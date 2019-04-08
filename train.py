import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
import argparse
import  model
import params


original_height = 960
original_width = 540

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