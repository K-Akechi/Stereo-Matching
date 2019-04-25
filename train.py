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
iterations = 60000

data_record = ["/home/new/Documents/Stereo-Matching/fly_train.tfrecords", "/home/new/Documents/Stereo-Matching/fly_test.tfrecords"]

train_dir = 'lr_consistency/'


def loss_func(left_input, right_input, disp_map_l):
    disp_map_l = disp_map_l / target_width
    # disp_map_r = disp_map_r / target_width
    left_reconstructed = bilinear_sampler_1d_h(right_input, -disp_map_l)
    # right_reconstructed = bilinear_sampler_1d_h(left_input, disp_map_r)
    # right_to_left_disp = bilinear_sampler_1d_h(disp_map_r, -disp_map_l)
    # left_to_right_disp = bilinear_sampler_1d_h(disp_map_l, disp_map_r)
#    print("reconstruct ok")
    left_wlcn = tf.abs(wlcn(left_input, left_reconstructed))
    # right_wlcn = tf.abs(wlcn(right_input, right_reconstructed))
    # left_second_wlcn = tf.abs(wlcn(left_input, left_second_warp))
    # right_second_wlcn = tf.abs(wlcn(right_input, right_second_warp))
    # count = 0
    # left_input = tf.pad(left_input, [[0, 0], [16, 16], [16, 16], [0, 0]])
    # left_wlcn = tf.pad(left_wlcn, [[0, 0], [16, 16], [16, 16], [0, 0]])
    # left_input_weights = []
    # left_input_sub = []
    # left_wlcn_slices = []
    # for b in range(batch_size):
    #     for row in range(target_height):
    #         for col in range(target_width):
    #             left_input_weights.append(left_input[b, row:row+32, col:col+32, :])
    #             left_input_sub.append(left_input[b, row+16, col+16, :])
    #             # adaptive_weight = tf.exp(-tf.abs(left_input[b, row:row+32, col:col+32, :] - left_input[b, row+16, col+16, :]) / 2.)
    #             # adaptive_weights.append(adaptive_weight)
    #             left_wlcn_slices.append(left_wlcn[b, row:row+32, col:col+32, :])
    #             # adaptive_cost = tf.reduce_sum(tf.abs(adaptive_weights * left_wlcn[b, row:row+32, col:col+32, :])) / tf.reduce_sum(adaptive_weights)
    #             # loss += adaptive_cost
    #             count += 1
    #             print count
    # # adaptive_weights = tf.stack(adaptive_weights, axis=0)
    # left_input_weights = tf.stack(left_input_weights, axis=0)
    # left_input_sub = tf.stack(left_input_sub, axis=0)
    # left_input_sub = tf.expand_dims(left_input_sub, axis=1)
    # left_input_sub = tf.expand_dims(left_input_sub, axis=1)
    # left_wlcn_slices = tf.stack(left_wlcn_slices, axis=0)
    # adaptive_weights = tf.exp(-tf.abs(left_input_weights - left_input_sub) / 2.)
    # print 'loss generated'
    # adaptive_cost = tf.reduce_sum(tf.abs(adaptive_weights * left_wlcn_slices), axis=(1, 2, 3)) / tf.reduce_sum(adaptive_weights, axis=(1, 2, 3))
    # loss = tf.reduce_sum(adaptive_cost)
   # left_wlcn = tf.abs(left_input - left_reconstructed)
   # print (left_wlcn)
   # print("wlcn ok")
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((33, 33, p.channels, 3)))
   #  left_wlcn_loss = tf.nn.conv2d(left_wlcn, guassian_filter, [1, 1, 1, 1], padding='SAME')
   #  right_wlcn_loss = tf.nn.conv2d(right_wlcn, guassian_filter, [1, 1, 1, 1], padding='SAME')
   #  left_second_wlcn_loss = tf.nn.conv2d(left_second_wlcn, guassian_filter, [1, 1, 1, 1], padding='SAME')
   #  right_second_wlcn_loss = tf.nn.conv2d(right_second_wlcn, guassian_filter, [1, 1, 1, 1], padding='SAME')
    left_wlcn_slices = tf.unstack(left_wlcn, axis=0)
    left_loss = []
    shape = left_wlcn_slices[0].get_shape().as_list()
    for item in left_wlcn_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_wlcn_slice = tf.nn.conv2d(item, guassian_filter, [1, 1, 1, 1], padding='SAME')
        left_wlcn_slice = tf.squeeze(left_wlcn_slice, axis=0)
        # left_wlcn_slice = (left_wlcn_slice - tf.reduce_min(left_wlcn_slice)) / (tf.reduce_max(left_wlcn_slice) - tf.reduce_min(left_wlcn_slice))
        left_loss.append(left_wlcn_slice)
    left_loss = tf.stack(left_loss)

    # right_wlcn_slices = tf.unstack(right_wlcn, axis=0)
    # right_loss = []
    # shape = right_wlcn_slices[0].get_shape().as_list()
    # for item in right_wlcn_slices:
    #     item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
    #     right_wlcn_slice = tf.nn.conv2d(item, guassian_filter, [1, 1, 1, 1], padding='SAME')
    #     right_wlcn_slice = tf.squeeze(right_wlcn_slice, axis=0)
    #     # left_wlcn_slice = (left_wlcn_slice - tf.reduce_min(left_wlcn_slice)) / (tf.reduce_max(left_wlcn_slice) - tf.reduce_min(left_wlcn_slice))
    #     right_loss.append(right_wlcn_slice)
    # right_loss = tf.stack(right_loss)
#    print(loss)
#     lr_left_loss = tf.reduce_mean(tf.abs(right_to_left_disp - disp_map_l))
#     lr_right_loss = tf.reduce_mean(tf.abs(left_to_right_disp - disp_map_r))
    return tf.reduce_mean(left_loss)


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
        # left_lcn_slice = (left_lcn_slice - tf.reduce_min(left_lcn_slice)) / (tf.reduce_max(left_lcn_slice) - tf.reduce_min(left_lcn_slice))
        left_lcn.append(left_lcn_slice)
    left_lcn = tf.stack(left_lcn)

    for item in left_rc_slices:
        item = tf.reshape(item, [1, shape[0], shape[1], shape[2]])
        left_rc_lcn_slice = util.LocalContrastNorm(item, radius=9)
        left_rc_lcn_slice = tf.squeeze(left_rc_lcn_slice, axis=0)
        # left_rc_lcn_slice = (left_rc_lcn_slice - tf.reduce_min(left_rc_lcn_slice)) / (tf.reduce_max(left_rc_lcn_slice) - tf.reduce_min(left_rc_lcn_slice))
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
        s_deviation = tf.pow(sum_square_loss + 1e-8, 0.5)
        # s_deviation = (s_deviation - tf.reduce_min(s_deviation)) / (tf.reduce_max(s_deviation) - tf.reduce_min(s_deviation))
        loss_s_deviation.append(s_deviation)
    tf.stack(loss_s_deviation)
    return loss


def unary_loss(left_input, right_input, disp_map_l, disp_map_r):
    shape = left_input.get_shape().as_list()
    left_warp = bilinear_sampler_1d_h(right_input, -disp_map_l)
    right_warp = bilinear_sampler_1d_h(left_input, disp_map_r)
    # guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, p.channels, 3)))
    #
    # left_lcn = tf.nn.conv2d(left_input, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_warp_lcn = tf.nn.conv2d(left_warp, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_lcn = tf.nn.conv2d(right_input, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_warp_lcn = tf.nn.conv2d(right_warp, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # # shape = left_input_slices[0].get_shape().as_list()
    #
    # left_lcn = left_input - left_lcn
    # left_square = tf.nn.conv2d(tf.square(left_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_deviation = tf.pow(left_square + 1e-8, 0.5)
    # left_lcn = left_lcn / (left_deviation + 1e-4)
    #
    # left_warp_lcn = left_warp - left_warp_lcn
    # left_warp_square = tf.nn.conv2d(tf.square(left_warp_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_warp_deviation = tf.pow(left_warp_square + 1e-8, 0.5)
    # left_warp_lcn = left_warp_lcn / (left_warp_deviation + 1e-4)
    #
    # right_lcn = right_input - right_lcn
    # right_square = tf.nn.conv2d(tf.square(right_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_deviation = tf.pow(right_square + 1e-8, 0.5)
    # right_lcn = right_lcn / (right_deviation + 1e-4)
    #
    # right_warp_lcn = right_warp - right_warp_lcn
    # right_warp_square = tf.nn.conv2d(tf.square(right_warp_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_warp_deviation = tf.pow(right_warp_square + 1e-8, 0.5)
    # right_warp_lcn = right_warp_lcn / (right_warp_deviation + 1e-4)

#    left_wlcn = left_lcn - left_warp_lcn

#    right_warp = bilinear_sampler_1d_h(left_input, -disp_map)
    N = shape[1]*shape[2]
    lambda1 = 0.80
    lambda2 = 0.15
    lambda3 = 0.15
    SSIM_loss = lambda1 * (tf.reduce_sum((1.0 - util.SSIM(left_input, left_warp)) / 2.0) + tf.reduce_sum((1.0 - util.SSIM(right_input, right_warp)) / 2.0))
#    LWCN_loss = lambda1 * tf.reduce_sum(tf.abs(left_wlcn))
    l1_loss = lambda2 * (tf.reduce_sum(tf.abs(left_input - left_warp)) + tf.reduce_sum(tf.abs(right_input - right_warp)))
    gradient_loss = lambda3 * (tf.reduce_sum(tf.abs(util.gradient_total(left_input) - util.gradient_total(left_warp))) +
                               tf.reduce_sum(tf.abs(util.gradient_total(right_input) - util.gradient_total(right_warp))))

    return (SSIM_loss + l1_loss + gradient_loss) / N


def regularization_loss(left_input, right_input, disp_map_l, disp_map_r):
    shape = left_input.get_shape().as_list()
    N = shape[1] * shape[2]
    second_grad_dx, second_grad_dy = util.second_gradient(disp_map_l)
    second_left_dx, second_left_dy = util.second_gradient(left_input)
    second_grad_r_dx, second_grad_r_dy = util.second_gradient(disp_map_r)
    second_right_dx, second_right_dy = util.second_gradient(right_input)
    reg_loss_l = tf.reduce_sum(tf.abs(second_grad_dx) * tf.exp(-tf.abs(second_left_dx)) + tf.abs(second_grad_dy) * tf.exp(-tf.abs(second_left_dy)))
    reg_loss_r = tf.reduce_sum(
        tf.abs(second_grad_r_dx) * tf.exp(-tf.abs(second_right_dx)) + tf.abs(second_grad_r_dy) * tf.exp(
            -tf.abs(second_right_dy)))
    return (reg_loss_l + reg_loss_r) / N


def consistency_loss(left_input, right_input, disp_map_l, disp_map_r):
    left_warp = bilinear_sampler_1d_h(right_input, -disp_map_l)
    right_warp_2 = bilinear_sampler_1d_h(left_warp, disp_map_r)
    right_warp = bilinear_sampler_1d_h(left_input, disp_map_r)
    left_warp_2 = bilinear_sampler_1d_h(right_warp, -disp_map_l)
    guassian_filter = tf.convert_to_tensor(util.get_gaussian_filter((9, 9, p.channels, 3)))

    # left_lcn = tf.nn.conv2d(left_input, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_warp_2_lcn = tf.nn.conv2d(left_warp_2, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_lcn = tf.nn.conv2d(right_input, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_warp_2_lcn = tf.nn.conv2d(right_warp_2, guassian_filter, [1, 1, 1, 1], padding='SAME')
    # # shape = left_input_slices[0].get_shape().as_list()
    #
    # left_lcn = left_input - left_lcn
    # left_square = tf.nn.conv2d(tf.square(left_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_deviation = tf.pow(left_square + 1e-8, 0.5)
    # left_lcn = left_lcn / (left_deviation + 1e-4)
    #
    # left_warp_2_lcn = left_warp_2 - left_warp_2_lcn
    # left_warp_square = tf.nn.conv2d(tf.square(left_warp_2_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # left_warp_deviation = tf.pow(left_warp_square + 1e-8, 0.5)
    # left_warp_2_lcn = left_warp_2_lcn / (left_warp_deviation + 1e-4)
    #
    # right_lcn = right_input - right_lcn
    # right_square = tf.nn.conv2d(tf.square(right_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_deviation = tf.pow(right_square + 1e-8, 0.5)
    # right_lcn = right_lcn / (right_deviation + 1e-4)
    #
    # right_warp_2_lcn = right_warp_2 - right_warp_2_lcn
    # right_warp_square = tf.nn.conv2d(tf.square(right_warp_2_lcn), guassian_filter, [1, 1, 1, 1], padding='SAME')
    # right_warp_deviation = tf.pow(right_warp_square + 1e-8, 0.5)
    # right_warp_2_lcn = right_warp_2_lcn / (right_warp_deviation + 1e-4)

    left_loss = tf.reduce_mean(tf.abs(left_input - left_warp_2))
    right_loss = tf.reduce_mean(tf.abs(right_input - right_warp_2))
    return left_loss + right_loss


def MDH_loss(disp_map_l, disp_map_r):
    return tf.reduce_mean(tf.abs(disp_map_l) + tf.abs(disp_map_r))


def loss_func_v1(left_input, right_input, pred_disp):
    pred_disp = pred_disp / target_width
    left_warp = bilinear_sampler_1d_h(right_input, -pred_disp)

    return tf.reduce_sum(tf.abs(left_input - left_warp))


def main(argv=None):
#    tf.logging.set_verbosity(tf.logging.ERROR)
#     tf.reset_default_graph()
    batch_train = util.read_and_decode(data_record[0])
    batch_test = util.read_and_decode(data_record[1])

    left_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    right_image = tf.placeholder(tf.float32, [batch_size, target_height, target_width, 3])
    phase = tf.placeholder(tf.bool)

    pred_disp_l = model.stereonet(left_image, right_image)

#    pred_disp_r = pred_disp_r / target_width
#    loss_graph = loss_func(left_image, right_image, scaled_disp)
#    loss = unary_loss(left_image, right_image, pred_disp_l, pred_disp_r) + consistency_loss(left_image, right_image, pred_disp_l, pred_disp_r) + 1e-3 * regularization_loss(left_image, right_image, pred_disp_l, pred_disp_r) + 1e-3 * MDH_loss(pred_disp_l, pred_disp_r)
    loss = loss_func(left_image, right_image, pred_disp_l)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # print regularization_losses
    #loss = tf.add_n([loss] + regularization_losses)
    avg_loss = tf.reduce_mean(loss)
#    loss = loss_func_v1(left_image, right_image, pred_disp_l)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY, global_step)
    # keep moving average for loss
    tf.add_to_collection(model.UPDATE_OPS_COLLECTION, ema.apply([loss]))

    batchnorm_updates = tf.get_collection(model.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    tf.summary.scalar('loss: ', loss)
    learning_rate = initial_lr
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='siamese|3d_conv|refinement')
    grads = optimizer.compute_gradients(loss, var_list=weights)
#    grads, variables = zip(*optimizer.compute_gradients(loss, var_list=weights))
#    grads, global_norm = tf.clip_by_global_norm(grads, 5)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    train = tf.group(train_op, batchnorm_updates_op)

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
            batch_tr = sess.run(batch_train)
#            print(batch)
            feed_dict = {left_image: batch_tr[0], right_image: batch_tr[1], phase: True}
#            print '1.'
#            disp = sess.run(pred_disp, feed_dict=feed_dict)
#            print disp
#            print '2.'
#            gradient = sess.run(grads, feed_dict=feed_dict)
#            print gradient
#            plt.imshow(scaled_disp.eval(session=sess))
#            plt.show()
            summary, _, loss_value, glb_step = sess.run([summary_op, train, avg_loss, global_step], feed_dict=feed_dict)
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
            if glb_step % 100 == 0 and step > 0:
                batch_te = sess.run(batch_test)
                feed_dict_t = {left_image: batch_te[0], right_image: batch_te[1], phase: True}
                loss_t, _ = sess.run([loss, tf.no_op()], feed_dict=feed_dict_t)
                print('Test loss = %.3f' % loss_t)
            if glb_step % 1000 == 0 and step > 0:
                saver.save(sess, train_dir, global_step=global_step)
        coord.request_stop()
        coord.join(threads)
    return


if __name__ == "__main__":
    tf.app.run()
