import tensorflow as tf
import numpy as np
import train

batch_size = train.batch_size
disparity_range = (train.max_disparity + 1) // pow(2, 3)
height = train.target_height
width = train.target_width


def residual_block(image, channels, stride, dilated):
    layer1 = tf.layers.conv2d(image, filters=channels, kernel_size=3, padding='same', strides=stride, dilation_rate=dilated)
    layer1 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer1))
    layer2 = tf.layers.conv2d(layer1, filters=channels, kernel_size=3, padding='same', strides=stride, dilation_rate=dilated)
    return tf.nn.leaky_relu(tf.layers.batch_normalization(image + layer2))


def siamese_network(image):
    layer = image
    layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=1)
    for i in range(3):
        layer = residual_block(layer, 32, 1, 1)
    for i in range(3):
        layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=2)
        layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=1)
    return layer


'''
def cost_volumn(left_features, right_features, method="subtract"):
    cost_volumn_list = []
    for disp in range((disparity_range + 1) // 8):
        paddings = [[0, 0], [0, 0], [disp, 0], [0, 0]]
        for k in range(32):
            left_feature = tf.slice(left_features,  [0, 0, 0, k], [height//8, width//8, 1])
            right_feature = tf.slice(right_features, [0, 0, 0, k], [height//8, width//8, 1])
            right_feature = tf.slice(right_feature, [0, 0, disp, 0], [height//8, width//8 - disp, 1])
            right_feature = tf.pad(right_feature, paddings, "CONSTANT")
#            cost_volumn_list.append(left_feature)
            if method == "subtract":
                cost_volumn_list.append(left_feature - right_feature)
            else:
                cost_volumn_list.append(left_feature)
                cost_volumn_list.append(right_feature)
    cost_volumn_list = tf.stack(cost_volumn_list)
    cost_volumn_list = tf.reshape(cost_volumn_list, shape=(batch_size, (disparity_range+1)//8, 32, height//8, width//8))
    cost_volumn_list = tf.transpose(cost_volumn_list, [0, 1, 3, 4, 2])

    return cost_volumn_list

def loss_fun(left_image, right_image, disparity_map):
    reconstruction_left = np.zeros(batch_size, height, width, 3)

    return
'''


def cost_volume(left_image, right_image):
    cost_volume_list = []
    constant_disp_shape = right_image.get_shape().as_list()#返回张量的shape

    for disp in range(disparity_range):
        right_moved = image_bias_move_v2(right_image, tf.constant(disp, dtype=tf.float32, shape=constant_disp_shape))
        tf.summary.image('right_siamese_moved', right_moved[:, :, :, :3], 2)
        cost_volume_list.append(tf.concat([left_image, right_moved], axis=-1))
    cost_volume = tf.stack(cost_volume_list, axis=1)

    for i in range(4):
        cost_volume = tf.layers.conv3d(cost_volume, filters=32, kernel_size=3, padding='same', strides=1)
        cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.9)

    disparity_volume = tf.reshape(tf.tile(tf.expand_dims(tf.range(disparity_range), axis=1), [1, constant_disp_shape[1] * constant_disp_shape[2] * cost_volume.get_shape().as_list()[-1]]), [1, -1])
    disparity_volume = tf.reshape(tf.tile(disparity_volume, [batch_size, 1]), [-1] + cost_volume.get_shape().as_list()[1: ])

    new_batch_slice = []
    batch_slice = tf.unstack(cost_volume, axis=0)
    for item in batch_slice:
        new_batch_slice.append(tf.nn.softmax(-item, axis=0))

    return tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice, axis=0), axis=1)


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
    biased_batch_2d_arr = tf.clip_by_value(tf.to_float(each_batch_2d_arr - disparity_map[:, :, :, 0]), 0., tf.to_float(image.get_shape()[2] - 1))

    # set start index for each batch and row
    initial_arr = tf.tile(tf.expand_dims(tf.range(image.get_shape()[1] * batch_size) * image.get_shape()[2], axis=-1), [1, image.get_shape()[2]])

    # finally add together without channels dim
    biased_batch_2d_arr_high = tf.clip_by_value(tf.to_int32(tf.floor(biased_batch_2d_arr + 1)), 0, image.get_shape()[2] - 1)
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
    with tf.variable_scope('first_part', reuse=tf.AUTO_REUSE):
        left_siamese = siamese_network(image_l)
        right_siamese = siamese_network(image_r)

    with tf.variable_scope('second_part', reuse=tf.AUTO_REUSE):
        disp_map = cost_volume(left_siamese, right_siamese)
        new_shape = disp_map.get_shape().as_list()
        new_shape[1] *= 8
        new_shape[2] *= 8
        disp_map = tf.image.resize_images(disp_map, [new_shape[1], new_shape[2]])

    with tf.variable_scope('third_part', reuse=tf.AUTO_REUSE):
        input_left = tf.layers.conv2d(image_l, filters=16, kernel_size=3, strides=1, padding='same')
        input_left = tf.nn.leaky_relu(tf.layers.batch_normalization(input_left))
        input_left = residual_block(input_left, 16, 1, 1)
        input_left = residual_block(input_left, 16, 1, 2)

        disp_map = tf.layers.conv2d(disp_map, filters=16, kernel_size=3, strides=1, padding='same')
        disp_map = tf.nn.leaky_relu(tf.layers.batch_normalization(disp_map))
        disp_map = residual_block(disp_map, 16, 1, 1)
        disp_map = residual_block(disp_map, 16, 1, 2)

        layer = tf.concat([input_left, disp_map], axis=-1)
        layer = residual_block(layer, 32, 1, 4)
        layer = residual_block(layer, 32, 1, 8)

        for i in range(2):
            layer = residual_block(layer, 32, 1, 1)

        disp_res = tf.layers.conv2d(layer, filters=1, kernel_size=3, strides=1, padding='same')

    return tf.add(disp_map, disp_res)


def invalidation_network(left_siamese, right_siamese, fullres_disp, left_input):
    layer = tf.concat([left_siamese, right_siamese], axis=-1)
    for i in range(5):
        layer = residual_block(layer, 64, 1, 1)
    layer = tf.layers.conv2d(layer, filters=1, kernel_size=3, strides=1, padding='same')
    new_shape = layer.get_shape().as_list()
    new_shape[1] *= 8
    new_shape[2] *= 8
    upsampled_invalid = tf.image.resize_images(layer, [new_shape[1], new_shape[2]])

    layer2 = tf.concat([upsampled_invalid, fullres_disp, left_input], axis=-1)
    layer2 = tf.layers.conv2d(layer2, filters=32, kernel_size=3, strides=1, padding='same')
    layer2 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer2))
    for i in range(4):
        layer2 = residual_block(layer2, 32, 1, 1)
    invalid_res = tf.layers.conv2d(layer2, filters=1, kernel_size=3, strides=1, padding='same')

    return tf.add(upsampled_invalid, invalid_res)
