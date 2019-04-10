import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
import argparse
import model
import math


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
    result = []

    left_slice = tf.unstack(left, axis=0)
    for item in left_slice:
        item = tf.squeeze(item)
        shape = item.get_shape().as_list()
        left_lcn = np.zeros(shape)
        item = tf.pad(item, [[4, 4], [4, 4]])
        for row in range(4, shape[0]+4):
            for col in range(4, shape[1]+4):
                patch = tf.slice(item, [row-4, col-4], [9, 9])
                mean, variance = tf.nn.moments(patch, [0, 1])
                standard_d = math.sqrt(variance)
                lcn_value = (item[row, col] - mean) / standard_d + 0.001

    return