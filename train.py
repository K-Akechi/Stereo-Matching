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
    optimized_loss = wb_optimization(left_input, left_wlcn)

    return optimized_loss


# Weighted Local Contrast Normalization
def wlcn(left, left_rc):
    #result = []

    left_slice = tf.unstack(left, axis=0)
    left_rc_slice = tf.unstack(left_rc, axis=0)
    shape = tf.squeeze(left_slice[0]).get_shape().as_list()
    left_lcn = []
    left_rc_lcn = []


    for item in left_slice:
        item = tf.squeeze(item)
        item = tf.pad(item, [[4, 4], [4, 4]])
        left_lcn_slice = tf.Variable(tf.zeros(shape))
        for row in range(4, shape[0]+4):
            for col in range(4, shape[1]+4):
                patch = tf.slice(item, [row-4, col-4], [9, 9])
                mean, variance = tf.nn.moments(patch, [0, 1])
                standard_d = tf.sqrt(variance)
                lcn_value = (item[row, col] - mean) / standard_d + 0.0001
                one_hot = util.get_one_hot_matrix(shape[0], shape[1], [row-4, col-4])
                left_lcn_slice = left_lcn_slice + one_hot * lcn_value
        left_lcn.append(left_lcn_slice)

    for item in left_rc_slice:
        item = tf.squeeze(item)
        item = tf.pad(item, [[4, 4], [4, 4]])
        left_rc_lcn_slice = tf.Variable(tf.zeros(shape))
        for row in range(4, shape[0]+4):
            for col in range(4, shape[1]+4):
                patch = tf.slice(item, [row-4, col-4], [9, 9])
                mean, variance = tf.nn.moments(patch, [0, 1])
                standard_d = tf.sqrt(variance)
                lcn_value = (item[row, col] - mean) / standard_d + 0.0001
                one_hot = util.get_one_hot_matrix(shape[0], shape[1], [row - 4, col - 4])
                left_rc_lcn_slice = left_rc_lcn_slice + one_hot * lcn_value
        left_rc_lcn.append(left_rc_lcn_slice)

    left_lcn = tf.stack(left_lcn)
    left_rc_lcn = tf.stack(left_rc_lcn)
    loss = left_lcn - left_rc_lcn
    loss_slice = tf.unstack(loss, axis=0)
    loss_wlcn = []
    for item in loss_slice:
        item = tf.squeeze(item)
        item = tf.pad(item, [[4, 4], [4, 4]])
        loss_sd_slice = tf.Variable(tf.zeros(shape))
        for row in range(4, shape[0]+4):
            for col in range(4, shape[1]+4):
                patch = tf.slice(item, [row-4, col-4], [9, 9])
                mean, variance = tf.nn.moments(patch, [0, 1])
                standard_d = tf.sqrt(variance)
                one_hot = util.get_one_hot_matrix(shape[0], shape[1], [row - 4, col - 4])
                loss_sd_slice = loss_sd_slice + one_hot * standard_d
        loss_wlcn.append(loss_sd_slice)

    loss_wlcn = tf.stack(loss_wlcn)
    return tf.abs(loss_wlcn)


# window-based optimization
def wb_optimization(left, wlcn):

    return









if __name__ == "__main__":
    tf.app.run()