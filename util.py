import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import params
import pdb


def get_gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float32')
    sum = 1

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[1] / 2.)
    for out_idx in range(0, kernel_shape[3]):
        for kernel_idx in range(0, kernel_shape[2]):
            for i in range(0, kernel_shape[0]):
                for j in range(0, kernel_shape[1]):
                    x[i, j, kernel_idx, out_idx] = gauss(i - mid, j - mid)
                    if out_idx == 0:
                        sum = np.sum(x)

    return x / sum


def LocalContrastNorm(image, radius=9):
    """
    image: tf.Tensor , .shape => (1, height, width, channels)

    radius: Gaussian filter size (int), odd
    """
    if radius % 2 == 0:
        radius += 1

    n, h, w, c = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    image = tf.pad(image, [[0, 0], [radius // 2, radius // 2], [radius // 2, radius // 2], [0, 0]])
    gaussian_filter = tf.convert_to_tensor(get_gaussian_filter((radius, radius, c, 1)))
    filtered_out = tf.nn.conv2d(image, gaussian_filter, [1, 1, 1, 1], padding='SAME')
    mid = int(np.floor(gaussian_filter.get_shape().as_list()[0] / 2.))

    ### Subtractive Normalization
    centered_image = image - filtered_out
    centered_image_origin_size = centered_image[:, mid:-mid, mid:-mid, :]
    ## Variance Calc
    sum_sqr_image = tf.nn.conv2d(tf.square(centered_image), gaussian_filter, [1, 1, 1, 1], padding='SAME')
    s_deviation = tf.sqrt(sum_sqr_image[:, mid:-mid, mid:-mid, :])
    # per_img_mean = tf.reduce_mean(s_deviation)

    ## Divisive Normalization

    divisor = s_deviation + 1e-4
    new_image = centered_image_origin_size / divisor
    return new_image


def read_and_decode(filename):
    p = params.Params()
    width, height = p.original_w, p.original_h
    batch_size = p.batch_size
    target_w, target_h = p.target_w, p.target_h

    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,

        features={
            'img_left': tf.FixedLenFeature([], tf.string),
            'img_right': tf.FixedLenFeature([], tf.string)
            #            'disparity': tf.FixedLenFeature([], tf.string)
        })

    image_left = tf.decode_raw(features['img_left'], tf.uint8)
    image_left = tf.reshape(image_left, [height, width, 3])
    
    image_right = tf.decode_raw(features['img_right'], tf.uint8)
    image_right = tf.reshape(image_right, [height, width, 3])


    #    disparity = tf.decode_raw(features['disparity'], tf.float32)
    #    disparity = tf.reshape(disparity, [height, width, 1])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_left = tf.cast(image_left, tf.float32) * (1. / 255)
    image_right = tf.cast(image_right, tf.float32) * (1. / 255)
#    print(image_left)
    concat = tf.concat([image_left, image_right], 2)
    img_crop = tf.random_crop(concat, [target_h, target_w, 6])
#    print(img_crop[:, :, 0:3], img_crop[:, :, 3:])
    image_left_batch, image_right_batch = tf.train.shuffle_batch(
        [img_crop[:, :, 0:3], img_crop[:, :, 3:]],
        batch_size=batch_size, capacity=50,
        min_after_dequeue=10, num_threads=2)

    return [image_left_batch, image_right_batch]


def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx


def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


def gradient_total(img):
    img_gradient_x = gradient_x(img)
    img_gradient_y = gradient_y(img)
    gradient = tf.reduce_mean(tf.abs(img_gradient_x), axis=3, keep_dims=True) + tf.reduce_mean(tf.abs(img_gradient_y), axis=3, keep_dims=True)
    return gradient

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

