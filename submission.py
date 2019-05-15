import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model
import util
import time
import os
from bilinear_sampler import *
height = 720
width = 1280
train_dir = 'realsense/'

left_f = np.sqrt(1362.27 ** 2 + 1361.38 ** 2)
right_f = np.sqrt(1363.04 ** 2 + 1362.71 ** 2)

baseline = np.sqrt(54.9842 ** 2 + 0.0495804 ** 2 + 0.247789 ** 2)

left_image = tf.placeholder(tf.float32, [1, height, width, 3])
right_image = tf.placeholder(tf.float32, [1, height, width, 3])

_, _, pred_disp_l, pred_disp_r = model.stereonet(left_image, right_image)

scaled_disp_l = pred_disp_l / width
left_warp = bilinear_sampler_1d_h(right_image, -scaled_disp_l)
left_warp = (left_warp - tf.reduce_min(left_warp)) / (tf.reduce_max(left_warp) - tf.reduce_min(left_warp))
sess = tf.Session()
new_saver = tf.train.Saver()
new_saver.restore(sess, tf.train.latest_checkpoint(train_dir))

cwd = os.getcwd()
path = cwd + '/dataset/TEST/A/'

for folder in os.listdir(path):
    img_path = path + folder

    left_img_name = os.listdir(img_path + '/left/')[0]
    right_img_name = os.listdir(img_path + '/right/')[0]

    left_img_path = img_path + '/left/' + left_img_name
    right_img_path = img_path + '/right/' + right_img_name

    imgL = np.array(Image.open(left_img_path))
    imgL = imgL.astype(np.float32) * (1. / 255) * 2. - 1.
    imgL = imgL[:height, :width]
    imgL = np.expand_dims(imgL, axis=0)
    imgL = np.expand_dims(imgL, axis=-1)
    imgL = np.tile(imgL, (1, 1, 1, 3))

    imgR = np.array(Image.open(right_img_path))
    imgR = imgR.astype(np.float32) * (1. / 255.) * 2. - 1.
    imgR = imgR[:height, :width]
    imgR = np.expand_dims(imgR, axis=0)
    imgR = np.expand_dims(imgR, axis=-1)
    imgR = np.tile(imgR, (1, 1, 1, 3))

    start_time = time.time()
    output1, output2 = sess.run([pred_disp_l, pred_disp_r], feed_dict={left_image: imgL, right_image: imgR})
    duration = time.time() - start_time
    print('%.3f sec/image' % duration)
    print(sess.run([tf.reduce_min(pred_disp_l), tf.reduce_max(pred_disp_l)], feed_dict={left_image: imgL, right_image: imgR}))
    output1 = tf.squeeze(output1)
    output2 = tf.squeeze(output2)
    mask1 = tf.cast(output1 > 0, tf.bool)
    mask2 = tf.cast(output2 > 0, tf.bool)
    mask3 = tf.cast(output1 <= 144, tf.bool)
    mask4 = tf.cast(output2 <= 144, tf.bool)
    output1 = tf.where(mask1 & mask3, output1, tf.zeros_like(output1))
    output2 = tf.where(mask2 & mask4, output2, tf.zeros_like(output2))

    output1 = output1 / 144 * 255.
    output2 = output2 / 144 * 255.
    output1 = tf.cast(output1, tf.uint8)
    output2 = tf.cast(output2, tf.uint8)

    output1 = output1.eval(session=sess)
    output2 = output2.eval(session=sess)


    save_path = cwd + '/dataset/DISP/A/' + folder + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.imsave(save_path + 'left_color.png', output1, cmap='jet')
    plt.imsave(save_path + 'right_color.png', output2, cmap='jet')
    output1 = Image.fromarray(output1)
    output2 = Image.fromarray(output2)
    output1.save(save_path + 'left_disp.png')
    output2.save(save_path + 'right_disp.png')
