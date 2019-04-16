import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import model
import util
import time
import params
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
iterations = 2000

data_record = [" fly_train.tfrecords", "fly_test.tfrecords"]

train_dir = 'saved_model/'


def loss_fun(left_input, right_input, disp_map):
    left_reconstructed = model.image_bias_move_v2(right_input, disp_map)
    left_wlcn = wlcn(left_input, left_reconstructed)
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, left_wlcn.get_shape().as_list()[4], 1)))
    left_wlcn_slices = tf.unstack(left_wlcn, axis=0)
    loss = []
    for item in left_wlcn_slices:
        left_wlcn_slice = tf.nn.conv2d(item, guassian_filter, [1, 1, 1, 1], padding='SAME')
        loss.append(left_wlcn_slice)
    loss = tf.stack(loss)

    return loss


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
        left_lcn.append(left_lcn_slice)
    left_lcn = tf.stack(left_lcn)

    for item in left_rc_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_rc_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_rc_lcn.append(left_rc_lcn_slice)
    left_rc_lcn = tf.stack(left_rc_lcn)

    loss = left_lcn - left_rc_lcn
    loss_slices = tf.unstack(loss, axis=0)
    loss_s_deviation = []
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, shape[2], 1)))

    for item in loss_slices:
        sum_square_loss = tf.nn.conv2d(tf.square(item), guassian_filter, [1, 1, 1, 1], padding='SAME')
        s_deviation = tf.sqrt(sum_square_loss)
        loss_s_deviation.append(s_deviation)

    return tf.stack(loss_s_deviation)


def main(argv=None):
    tf.logging.set_verbosity(tf.logging.ERROR)

    batch_train = util.read_and_decode(data_record[0])
    batch_test = util.read_and_decode(data_record[1])

    left_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    right_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    phase = tf.placeholder(tf.bool)

    pred_disp = model.stereonet(left_image, right_image)

    loss = tf.reduce_mean(loss_fun(left_image, right_image, pred_disp))
    tf.summary.scalar('loss: ', loss)

    global_step = tf.Variable(0, name='globa_step', trainable=False)
    learning_rate = initial_lr
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='first_part|second_part|third_part')
    grads = optimizer.compute_gradients(loss, var_list=weights)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)


    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement = True)

    with tf.Session(config=config) as sess:
        restore_dir = tf.train.latest_checkpoint(train_dir)
        if restore_dir:
            saver.restore(sess, restore_dir)
            print('restore succeed')
        else:
            sess.run(init)
        summart_writer = tf.summary.FileWriter(train_dir+'log', sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(iterations+1):
            start_time = time.time()
            batch = sess.run(batch_train)
            print(batch)
            feed_dict = {left_image: batch[0], right_image: batch[1], phase: True}
            summary, _, loss_value, glb_step = sess.run([summary_op, train_op, loss, global_step], feed_dict=feed_dict)
            duration = time.time() - start_time
            write_summary = glb_step % 100
            if write_summary or step == 0:
                summart_writer.add_summary(summary)
            if glb_step % 2 == 0 and step > 0:
                print('Step %d: training loss = %.2f (%.3f sec/batch)' % (glb_step, loss_value, duration))
            if glb_step % 1000 == 0 and step > 0:
                saver.save(sess, train_dir, global_step=global_step)
        coord.request_stop()
        coord.join(threads)
    return


if __name__ == "__main__":
    tf.app.run()
