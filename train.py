import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
import argparse
import model


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

    return