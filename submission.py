import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model
import util
import time
from bilinear_sampler import *
height = 512
width = 960
train_dir = 'saved_model/'


left_image = tf.placeholder(tf.float32, [1, height, width, 3])
right_image = tf.placeholder(tf.float32, [1, height, width, 3])

pred_disp_l = model.stereonet(left_image, right_image)

scaled_disp_l = pred_disp_l / width
left_warp = bilinear_sampler_1d_h(right_image, -scaled_disp_l)
left_warp = (left_warp - tf.reduce_min(left_warp)) / (tf.reduce_max(left_warp) - tf.reduce_min(left_warp))
sess = tf.Session()
new_saver = tf.train.Saver()
new_saver.restore(sess, tf.train.latest_checkpoint(train_dir))

imgL = np.array(Image.open('/home/new/Documents/Stereo-Matching/0006_l.png'))
imgL = imgL.astype(np.float32) * (1. / 255)
imgL = imgL[:height, :width]
imgL = np.expand_dims(imgL, axis=0)


imgR = np.array(Image.open('/home/new/Documents/Stereo-Matching/0006_r.png'))
imgR = imgR.astype(np.float32) * (1. / 255)
imgR = imgR[:height, :width]
imgR = np.expand_dims(imgR, axis=0)
pred_disp_l = pred_disp_l / 192
start_time = time.time()
output = sess.run(pred_disp_l, feed_dict={left_image: imgL, right_image: imgR})
duration = time.time() - start_time
print(sess.run([tf.reduce_min(pred_disp_l), tf.reduce_max(pred_disp_l)], feed_dict={left_image: imgL, right_image: imgR}))
output = tf.squeeze(output)
plt.imshow(output.eval(session=sess), cmap='jet')
plt.show()
