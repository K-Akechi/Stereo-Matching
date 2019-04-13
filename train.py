import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
import argparse
import model
import util


original_height = 540
original_width = 960
target_height = 256
target_width = 512

max_disparity = 192
batch_size = 16

initial_lr = 1e-4
iterations = 2000


def loss_fun(left_input, right_input, disp_map):
    left_reconstructed = model.image_bias_move_v2(right_input, disp_map)
    left_wlcn = wlcn(left_input, left_reconstructed)


    return


# Weighted Local Contrast Normalization
def wlcn(left, left_rc):
    #result = []

    left_slice = tf.unstack(left, axis=0)
    left_rc_slice = tf.unstack(left_rc, axis=0)
    left_lcn = []
    left_rc_lcn = []

    shape = left_slice[0].get_shape().as_list()

    for item in left_slice:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_lcn.append(left_lcn_slice)
    left_lcn = tf.stack(left_lcn)

    for item in left_rc_slice:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_rc_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_rc_lcn.append(left_rc_lcn_slice)
    left_rc_lcn = tf.stack(left_rc_lcn)

    loss = left_lcn - left_rc_lcn
    loss_slice = tf.unstack(loss, axis=0)
    loss_s_deviation = []
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, shape[2])))

    for item in loss_slice:
        sum_square_loss = tf.nn.conv2d(tf.square(item), guassian_filter, [1, 1, 1, 1], padding='SAME')
        s_deviation = tf.sqrt(sum_square_loss)
        loss_s_deviation.append(s_deviation)

    return tf.stack(loss_s_deviation)


if __name__ == "__main__":
    tf.app.run()