import tensorflow as tf
import params


batch_size = params.batch_size
disparity_range = params.max_disparity

def residual_block(image):
    layer1 = tf.layers.conv2d(image, filters=32, kernel_size=3, padding='same')
    layer1 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer1))
    layer2 = tf.layers.conv2d(layer1, filters=32, kernel_size=3, padding='same')
    return tf.nn.leaky_relu(image + layer2)


def siamese_network(image):
    layer = image
    layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=1)
    for i in range(3):
        layer = residual_block(layer)
    for i in range(3):
        layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=2)
        layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, padding='same', strides=1)
    return layer


def cost_volume(left_image, right_image):
    cost_volume_list = []
    constant_disp_shape = right_image.get_shape().as_list()

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