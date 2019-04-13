import tensorflow as tf
import numpy as np


def get_gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[1] / 2.)
    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)

    return x / np.sum(x)


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
