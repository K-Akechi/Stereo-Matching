import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import model
import util
import time
import params
from bilinear_sampler import *
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = params.Params()
original_height = p.original_h
original_width = p.original_w
target_height = p.target_h
target_width = p.target_w

max_disparity = p.max_disparity
batch_size = p.batch_size

initial_lr = 1e-4
iterations = 20000

data_record = ["/home/new/Documents/Stereo-Matching/fly_train.tfrecords", "/home/new/Documents/Stereo-Matching/fly_test.tfrecords"]

train_dir = 'saved_model/'


def loss_func(left_input, right_input, disp_map):
    left_reconstructed = bilinear_sampler_1d_h(right_input, disp_map)
#    print("reconstruct ok")
#    left_wlcn = wlcn(left_input, left_reconstructed)
    left_wlcn = tf.abs(left_input - left_reconstructed)
#    print (left_wlcn)
#    print("wlcn ok")
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, p.channels, 3)))
    left_wlcn_slices = tf.unstack(left_wlcn, axis=0)
    loss = []
    shape = left_wlcn_slices[0].get_shape().as_list()
    for item in left_wlcn_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_wlcn_slice = tf.nn.conv2d(item, guassian_filter, [1, 1, 1, 1], padding='SAME')
        left_wlcn_slice = tf.squeeze(left_wlcn_slice, axis=0)
        loss.append(left_wlcn_slice)
    loss = tf.stack(loss)
#    print(loss)
    return tf.abs(loss)


# Weighted Local Contrast Normalization
def wlcn(left, left_rc):

    left_slices = tf.unstack(left, axis=0)
    left_rc_slices = tf.unstack(left_rc, axis=0)
    left_lcn = []
    left_rc_lcn = []

    shape = left_slices[0].get_shape().as_list()

    for item in left_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_lcn_slice = tf.squeeze(left_lcn_slice, axis=0)
        left_lcn.append(left_lcn_slice)
    left_lcn = tf.stack(left_lcn)

    for item in left_rc_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_rc_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_rc_lcn_slice = tf.squeeze(left_rc_lcn_slice, axis=0)
        left_rc_lcn.append(left_rc_lcn_slice)
    left_rc_lcn = tf.stack(left_rc_lcn)

    loss = tf.abs(left_lcn - left_rc_lcn)
#    print(loss)
    loss_slices = tf.unstack(loss, axis=0)
    loss_s_deviation = []
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, p.channels, 3)))

    for item in loss_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        sum_square_loss = tf.nn.conv2d(tf.square(item), guassian_filter, [1, 1, 1, 1], padding='SAME')
        sum_square_loss = tf.squeeze(sum_square_loss, axis=0)
        s_deviation = tf.sqrt(sum_square_loss)
        loss_s_deviation.append(s_deviation)

    return tf.stack(loss_s_deviation)


def loss_func_v1(left_input, right_input, disp_map):
    right_r = bilinear_sampler_1d_h(right_input, disp_map)

    left_r = bilinear_sampler_1d_h(left_input, -disp_map)
#    tf.summary.image('right_input', right_input, 1)
#    tf.summary.image('right_moved', right, 1)
#    tf.summary.image('disparity', disp_map, 1)
    l = tf.abs(right_r - left_input) + tf.abs(right_input - left_r)

    return l


def unary_loss(left_input, right_input, disp_map):
    shape = left_input.get_shape().as_list()
    left_warp = bilinear_sampler_1d_h(right_input, disp_map)
#    right_warp = bilinear_sampler_1d_h(left_input, -disp_map)
    N = shape[1]*shape[2]
    lambda1 = 0.80
    lambda2 = 0.15
    lambda3 = 0.15
    SSIM_loss = lambda1 * tf.reduce_sum((1.0 - util.SSIM(left_input, left_warp)) / 2.0) / N
    l1_loss = lambda2 * tf.reduce_sum(tf.abs(left_input - left_warp))
    gradient_loss = lambda3 * tf.reduce_sum(tf.abs(util.gradient_total(left_input) - util.gradient_total(left_warp)))

    return SSIM_loss + l1_loss + gradient_loss


def regularization_loss(left_input, disp_map):

    return


def main(argv=None):
#    tf.logging.set_verbosity(tf.logging.ERROR)

    batch_train = util.read_and_decode(data_record[0])
    batch_test = util.read_and_decode(data_record[1])

    left_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    right_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    phase = tf.placeholder(tf.bool)

    pred_disp = model.stereonet(left_image, right_image)
    scaled_disp = (pred_disp - tf.reduce_min(pred_disp)) / (tf.reduce_max(pred_disp) - tf.reduce_min(pred_disp))
#    loss_graph = loss_func(left_image, right_image, scaled_disp)
    loss = tf.reduce_mean(loss_func(left_image, right_image, scaled_disp))
    tf.summary.scalar('loss: ', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = initial_lr
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='first_part|second_part|third_part')
    grads = optimizer.compute_gradients(loss, var_list=weights)
#    grads, variables = zip(*optimizer.compute_gradients(loss, var_list=weights))
#    grads, global_norm = tf.clip_by_global_norm(grads, 5)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)


    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        restore_dir = tf.train.latest_checkpoint(train_dir)
        if restore_dir:
            saver.restore(sess, restore_dir)
            print('restore succeed')
        else:
            sess.run(init)
        summary_writer = tf.summary.FileWriter(train_dir+'log', sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(iterations+1):
            start_time = time.time()
#            pdb.set_trace()
#            sess.graph.finalize()
            batch = sess.run(batch_train)
#            print(batch)
            feed_dict = {left_image: batch[0], right_image: batch[1], phase: True}
#            print '1.'
#            disp = sess.run(pred_disp, feed_dict=feed_dict)
#            print disp
#            print '2.'
#            gradient = sess.run(grads, feed_dict=feed_dict)
#            print gradient
#            plt.imshow(scaled_disp.eval(session=sess))
#            plt.show()
            summary, _, loss_value, glb_step = sess.run([summary_op, train_op, loss, global_step], feed_dict=feed_dict)
#            loss_g, loss_value, glb_step = sess.run([loss_graph, loss, global_step], feed_dict=feed_dict)
#            print(loss_g)
#            print '3.'
#            print(sess.run(grads, feed_dict=feed_dict))
#            sess.run(train_op, feed_dict=feed_dict)
#            glb_step = sess.run(global_step, feed_dict=feed_dict)
            duration = time.time() - start_time
            write_summary = glb_step % 100

            if write_summary or step == 0:
                summary_writer.add_summary(summary)

            if glb_step % 2 == 0 and step > 0:
                print('Step %d: training loss = %.3f (%.3f sec/batch)' % (glb_step, loss_value, duration))
            if glb_step % 1000 == 0 and step > 0:
                saver.save(sess, train_dir, global_step=global_step)
        coord.request_stop()
        coord.join(threads)
    return


if __name__ == "__main__":
    tf.app.run()
