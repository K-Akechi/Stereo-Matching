import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np
import params
import bilinear_sampler

p = params.Params()
MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00001
CONV_WEIGHT_STDDEV = 0.05
ASN_VARIABLES = 'asn_variables'
UPDATE_OPS_COLLECTION = 'gc_update_ops'
batch_size = p.batch_size
disparity_range = (p.max_disparity + 1) // pow(2, 3)
height = p.target_h
width = p.target_w


def conv2d(img, channels, stride, dilated):
    filters_in = img.get_shape()[-1]
    shape = [3, 3, filters_in, channels]
    weights = tf.get_variable('weights',
                              shape=shape,
                              dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                              trainable=True)
    bias = tf.get_variable('bias', [channels], 'float32', tf.constant_initializer(0.05, dtype='float'))
    if dilated > 1:
        img = tf.nn.atrous_conv2d(img, weights, dilated, padding='SAME')
    else:
        img = tf.nn.conv2d(img, weights, [1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(img, bias)


def conv3d(img, channels, stride, dilated):
    filters_in = img.get_shape()[-1]
    shape = [3, 3, 3, filters_in, channels]
    weights = tf.get_variable('weights',
                              shape=shape,
                              dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                              trainable=True)
    bias = tf.get_variable('bias', [channels], 'float32', tf.constant_initializer(0.05, dtype='float'))
    img = tf.nn.conv3d(img, weights, [1, stride, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(img, bias)


def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta',
                           shape=params_shape,
                           initializer=tf.zeros_initializer(),
                           dtype='float32',
                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                           trainable=True)
    gamma = tf.get_variable('gamma',
                            shape=params_shape,
                            initializer=tf.ones_initializer(),
                            dtype='float32',
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                            trainable=True)

    moving_mean = tf.get_variable('moving_mean',
                                  shape=params_shape,
                                  initializer=tf.zeros_initializer(),
                                  dtype='float32',
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                                  trainable=False)
    moving_variance = tf.get_variable('moving_variance',
                                      shape=params_shape,
                                      initializer=tf.ones_initializer(),
                                      dtype='float32',
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, ASN_VARIABLES],
                                      trainable=False)

    # These ops will only be performed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x


def residual_block(img, channels, stride, dilated, is_training):
    short_cut = img
    with tf.variable_scope('block_A'):
        img = conv2d(img, channels, stride, dilated)
        img = bn(img, is_training)
        img = tf.nn.leaky_relu(img)
    with tf.variable_scope('block_B'):
        img = conv2d(img, channels, stride, dilated)
        img = bn(img, is_training)
        img = short_cut + img
        img = tf.nn.leaky_relu(img)
    return img


def siamese_network(img, is_training):

    with tf.variable_scope('resnet'):
        with tf.variable_scope('first_layer'):
            img = conv2d(img, 32, 1, 1)
        for i in range(3):
            with tf.variable_scope('resnet' + str(i + 1)):
                img = residual_block(img, 32, 1, 1, is_training)

    with tf.variable_scope('downsample'):
        for i in range(3):
            with tf.variable_scope('2dconv' + str(i + 1)):
                img = conv2d(img, 32, 2, 1)
                img = bn(img, is_training)
        img = conv2d(img, 32, 1, 1)

    return img


def cost_volume(left_image, right_image):
    cost_volume_list = []
    constant_disp_shape = right_image.get_shape().as_list()
    #    print constant_disp_shape
    for disp in range(1, disparity_range + 1):
        right_moved = image_bias_move_v2(right_image, tf.constant(disp, dtype=tf.float32, shape=constant_disp_shape))
        tf.summary.image('right_siamese_moved', right_moved[:, :, :, :3], 2)
        # cost_volume_list.append(tf.concat([left_image, right_moved], axis=-1))
        cost_volume_list.append(left_image - right_moved)
        '''
        paddings = [[0, 0], [0, 0], [disp, 0], [0, 0]]
        for k in range(constant_disp_shape[3]):
            left_feature = tf.slice(left_image, [0, 0, 0, k], [batch_size, height/8, width/8, 1])
            right_feature = tf.slice(right_image, [0, 0, 0, k], [batch_size, height/8, width/8, 1])
            right_feature = tf.slice(right_feature, [0, 0, disp, 0], [batch_size, height/8, width/8 - disp, 1])
            right_feature = tf.pad(right_feature, paddings, "CONSTANT")
            cost_volume_list.append(tf.concat([left_feature, right_feature], axis=-1))
        '''
    cost_volume = tf.stack(cost_volume_list, axis=0)
    # cost_volume = tf.reshape(cost_volume, shape=(disparity_range, 2*constant_disp_shape[3], batch_size, height/8, width/8))
    cost_volume = tf.transpose(cost_volume, [1, 0, 2, 3, 4])

    # for i in range(4):
    #     cost_volume = tf.layers.conv3d(cost_volume, filters=32, kernel_size=3, padding='same', strides=1)
    #     cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    # cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    # #    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.9)
    #
    # disparity_volume = tf.reshape(tf.tile(tf.expand_dims(tf.range(1, disparity_range + 1), axis=1), [1,
    #                                                                                                  constant_disp_shape[
    #                                                                                                      1] *
    #                                                                                                  constant_disp_shape[
    #                                                                                                      2] *
    #                                                                                                  cost_volume.get_shape().as_list()[
    #                                                                                                      -1]]), [1, -1])
    # disparity_volume = tf.reshape(tf.tile(disparity_volume, [batch_size, 1]), cost_volume.get_shape().as_list())
    #
    # new_batch_slice = []
    # batch_slice = tf.unstack(cost_volume, axis=0)
    # for item in batch_slice:
    #     new_batch_slice.append(tf.nn.softmax(-item, axis=0))
    #
    # return tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice, axis=0), axis=1)
    return cost_volume


def cost_volume_v2(left_image, right_image):
    cost_volume_list = []
    constant_disp_shape = left_image.get_shape().as_list()
    #    print constant_disp_shape
    for disp in range(1, disparity_range + 1):
        left_moved = image_bias_move_v2(left_image, tf.constant(-disp, dtype=tf.float32, shape=constant_disp_shape))
        tf.summary.image('left_siamese_moved', left_moved[:, :, :, :3], 2)
        cost_volume_list.append(right_image - left_moved)
        '''
        paddings = [[0, 0], [0, 0], [disp, 0], [0, 0]]
        for k in range(constant_disp_shape[3]):
            left_feature = tf.slice(left_image, [0, 0, 0, k], [batch_size, height/8, width/8, 1])
            right_feature = tf.slice(right_image, [0, 0, 0, k], [batch_size, height/8, width/8, 1])
            right_feature = tf.slice(right_feature, [0, 0, disp, 0], [batch_size, height/8, width/8 - disp, 1])
            right_feature = tf.pad(right_feature, paddings, "CONSTANT")
            cost_volume_list.append(tf.concat([left_feature, right_feature], axis=-1))
        '''
    cost_volume = tf.stack(cost_volume_list, axis=0)
    #    cost_volume = tf.reshape(cost_volume, shape=(disparity_range, 2*constant_disp_shape[3], batch_size, height/8, width/8))
    cost_volume = tf.transpose(cost_volume, [1, 0, 2, 3, 4])
    # print cost_volume
    # for i in range(4):
    #     cost_volume = tf.layers.conv3d(cost_volume, filters=32, kernel_size=3, padding='same', strides=1)
    #     cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    # cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    # #    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.9)
    #
    # disparity_volume = tf.reshape(tf.tile(tf.expand_dims(tf.range(1, disparity_range + 1), axis=1), [1,
    #                                                                                                  constant_disp_shape[
    #                                                                                                      1] *
    #                                                                                                  constant_disp_shape[
    #                                                                                                      2] *
    #                                                                                                  cost_volume.get_shape().as_list()[
    #                                                                                                      -1]]), [1, -1])
    # disparity_volume = tf.reshape(tf.tile(disparity_volume, [batch_size, 1]), cost_volume.get_shape().as_list())
    #
    # new_batch_slice = []
    # batch_slice = tf.unstack(cost_volume, axis=0)
    # for item in batch_slice:
    #     new_batch_slice.append(tf.nn.softmax(-item, axis=0))
    #
    # return tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice, axis=0), axis=1)
    return cost_volume


def image_bias_move_v2(image, disparity_map):
    image = tf.pad(image, paddings=[[0, 0], [0, 0], [1, 1], [0, 0]])
    disparity_map = tf.pad(disparity_map, paddings=[[0, 0], [0, 0], [1, 1], [0, 0]])

    # create fundamental matrix
    each_1d_arr = tf.range(image.get_shape()[2])
    each_2d_arr = tf.tile(tf.expand_dims(each_1d_arr, axis=0), [image.get_shape()[1], 1])
    each_batch_2d_arr = tf.tile(tf.expand_dims(each_2d_arr, axis=0), [batch_size, 1, 1])
    each_batch_2d_arr = tf.to_float(each_batch_2d_arr)

    # sub/add bias value
    if len(disparity_map.get_shape().as_list()) == 3:
        tf.expand_dims(disparity_map, axis=-1)
    biased_batch_2d_arr = tf.clip_by_value(tf.to_float(each_batch_2d_arr - disparity_map[:, :, :, 0]), 0.,
                                           tf.to_float(image.get_shape()[2] - 1))

    # set start index for each batch and row
    initial_arr = tf.tile(tf.expand_dims(tf.range(image.get_shape()[1] * batch_size) * image.get_shape()[2], axis=-1),
                          [1, image.get_shape()[2]])

    # finally add together without channels dim
    biased_batch_2d_arr_high = tf.clip_by_value(tf.to_int32(tf.floor(biased_batch_2d_arr + 1)), 0,
                                                image.get_shape()[2] - 1)
    biased_batch_2d_arr_low = tf.clip_by_value(tf.to_int32(tf.floor(biased_batch_2d_arr)), 0, image.get_shape()[2] - 1)

    index_arr_high = tf.to_int32(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr_high, [-1])
    index_arr_low = tf.to_int32(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr_low, [-1])
    index_arr = tf.to_float(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr, [-1])

    weight_low = tf.to_float(index_arr_high) - tf.to_float(index_arr)
    weight_high = tf.to_float(index_arr) - tf.to_float(index_arr_low)

    # consider channel
    weight_low = tf.expand_dims(weight_low, axis=-1)
    weight_high = tf.expand_dims(weight_high, axis=-1)
    n_image = tf.reshape(image, [-1, image.get_shape()[-1]])
    new_image = weight_low * tf.gather(n_image, index_arr_low) + weight_high * tf.gather(n_image, index_arr_high)

    return tf.reshape(new_image, image.get_shape())[:, :, 1: -1, :]


def stereonet(image_l, image_r):
    is_training = tf.convert_to_tensor(p.is_training, dtype='bool', name='is_training')
    with tf.variable_scope('siamese') as scope:
        left_siamese = siamese_network(image_l, is_training)
        scope.reuse_variables()
        right_siamese = siamese_network(image_r, is_training)
        # print left_siamese
    constant_disp_shape = left_siamese.get_shape().as_list()
    left_cost_volumn = cost_volume(left_siamese, right_siamese)
    # right_cost_volumn = cost_volume_v2(left_siamese, right_siamese)

    with tf.variable_scope('3d_conv') as scope:
        for i in range(4):
            with tf.variable_scope('3d_conv' + str(i + 1)):
                left_cost_volumn = conv3d(left_cost_volumn, 32, 1, 1)
        with tf.variable_scope('3d_conv5'):
            left_cost_volumn = conv3d(left_cost_volumn, 1, 1, 1)
        # scope.reuse_variables()
        # for i in range(4):
        #     with tf.variable_scope('3d_conv' + str(i + 1)):
        #         right_cost_volumn = conv3d(right_cost_volumn, 32, 1, 1)
        # with tf.variable_scope('3d_conv5'):
        #     right_cost_volumn = conv3d(right_cost_volumn, 1, 1, 1)

        disparity_volume = tf.reshape(tf.tile(tf.expand_dims(tf.range(1, disparity_range + 1), axis=1), [1, constant_disp_shape[1] * constant_disp_shape[2] * left_cost_volumn.get_shape().as_list()[-1]]), [1, -1])
        disparity_volume = tf.reshape(tf.tile(disparity_volume, [batch_size, 1]), left_cost_volumn.get_shape().as_list())

        new_batch_slice_left = []
        # new_batch_slice_right = []
        batch_slice_left = tf.unstack(left_cost_volumn, axis=0)
        # batch_slice_right = tf.unstack(right_cost_volumn, axis=0)

        for item in batch_slice_left:
            new_batch_slice_left.append(tf.nn.softmax(-item, axis=0))
        # for item in batch_slice_right:
        #     new_batch_slice_right.append(tf.nn.softmax(-item, axis=0))

        disp_map_l = tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice_left, axis=0), axis=1)
        # disp_map_r = tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice_right, axis=0), axis=1)
        new_shape = disp_map_l.get_shape().as_list()
        new_shape[1] *= 8
        new_shape[2] *= 8
        disp_map_l_upsampled = tf.image.resize_images(disp_map_l, [new_shape[1], new_shape[2]])
        # disp_map_r_upsampled = tf.image.resize_images(disp_map_r, [new_shape[1], new_shape[2]])

    with tf.variable_scope('refinement') as scope:

        with tf.variable_scope('lbranch1'):
            input_left = conv2d(image_l, 16, 1, 1)
            input_left = bn(input_left, is_training)
            input_left = tf.nn.leaky_relu(input_left)
            with tf.variable_scope('res1'):
                input_left = residual_block(input_left, 16, 1, 1, is_training)
            with tf.variable_scope('res2'):
                input_left = residual_block(input_left, 16, 1, 2, is_training)

        with tf.variable_scope('lbranch2'):
            left_disp = conv2d(disp_map_l_upsampled, 16, 1, 1)
            left_disp = bn(left_disp, is_training)
            left_disp = tf.nn.leaky_relu(left_disp)
            with tf.variable_scope('res1'):
                left_disp = residual_block(left_disp, 16, 1, 1, is_training)
            with tf.variable_scope('res2'):
                left_disp = residual_block(left_disp, 16, 1, 2, is_training)

        with tf.variable_scope('lmerge'):
            layer = tf.concat([input_left, left_disp], axis=-1)
            with tf.variable_scope('res1'):
                layer = residual_block(layer, 32, 1, 4, is_training)
            with tf.variable_scope('res2'):
                layer = residual_block(layer, 32, 1, 8, is_training)
            for i in range(2):
                with tf.variable_scope('res' + str(i + 3)):
                    layer = residual_block(layer, 32, 1, 1, is_training)
            left_res = conv2d(layer, 1, 1, 1)

        # scope.reuse_variables()

        # with tf.variable_scope('rbranch1'):
        #     input_right = conv2d(image_r, 16, 1, 1)
        #     input_right = bn(input_right, is_training)
        #     input_right = tf.nn.leaky_relu(input_right)
        #     with tf.variable_scope('res1'):
        #         input_right = residual_block(input_right, 16, 1, 1, is_training)
        #     with tf.variable_scope('res2'):
        #         input_right = residual_block(input_right, 16, 1, 2, is_training)
        #
        # with tf.variable_scope('rbranch2'):
        #     right_disp = conv2d(disp_map_r_upsampled, 16, 1, 1)
        #     right_disp = bn(right_disp, is_training)
        #     right_disp = tf.nn.leaky_relu(right_disp)
        #     with tf.variable_scope('res1'):
        #         right_disp = residual_block(right_disp, 16, 1, 1, is_training)
        #     with tf.variable_scope('res2'):
        #         right_disp = residual_block(right_disp, 16, 1, 2, is_training)
        #
        # with tf.variable_scope('rmerge'):
        #     layer = tf.concat([input_right, right_disp], axis=-1)
        #     with tf.variable_scope('res1'):
        #         layer = residual_block(layer, 32, 1, 4, is_training)
        #     with tf.variable_scope('res2'):
        #         layer = residual_block(layer, 32, 1, 8, is_training)
        #     for i in range(2):
        #         with tf.variable_scope('res' + str(i + 3)):
        #             layer = residual_block(layer, 32, 1, 1, is_training)
        #     right_res = conv2d(layer, 1, 1, 1)

    return tf.add(disp_map_l_upsampled, left_res)
