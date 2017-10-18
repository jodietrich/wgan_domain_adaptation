# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
import numpy as np
from math import sqrt

def flatten(tensor):
    '''
    Flatten the last N-1 dimensions of a tensor only keeping the first one, which is typically 
    equal to the number of batches. 
    Example: A tensor of shape [10, 200, 200, 32] becomes [10, 1280000] 
    '''
    rhs_dim = get_rhs_dim(tensor)
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    '''
    Get the multiplied dimensions of the last N-1 dimensions of a tensor. 
    I.e. an input tensor with shape [10, 200, 200, 32] leads to an output of 1280000 
    '''
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])

def put_kernels_on_grid(images, pad=1, rescale_mode='automatic', input_range=None):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    Args:
      images:            [batch_size, X, Y, channels] 
      pad:               number of black pixels around each filter (between them)
      rescale_mode:      'manual' or 'automatic'
      Automatic rescale mode scales the images such that the they are in the range [0,255]
      Manual rescale mode maps input_range to [0,255] and thresholds everything outside the range
      input_range:       input range used for manual rescaling (min, max)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of images')
                return (i, int(n / i))

    (grid_Y, grid_X) = factorization(images.get_shape()[0].value)
    print('grid: %d = (%d, %d)' % (images.get_shape()[0].value, grid_Y, grid_X))

    if rescale_mode == 'automatic':
        x_min = tf.reduce_min(images)
        x_max = tf.reduce_max(images)
    elif rescale_mode == 'manual':
        x_min = input_range[0]
        x_max = input_range[1]

    images = (images - x_min) / (x_max - x_min)
    images = 255.0 * images
    if rescale_mode == 'manual':
        # threshold such that everything is in [0,255]
        tf.clip_by_value(images, 0, 255)


    # pad X and Y
    x = tf.pad(images, tf.constant([[0, 0], [pad, pad], [pad, pad],[0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = images.get_shape().as_list()[1] + 2 * pad
    X = images.get_shape().as_list()[2] + 2 * pad

    channels = images.get_shape()[3]

    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))

    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # Transpose the image again
    x = tf.transpose(x, (0, 2, 1, 3))

    return x

def put_kernels_on_grid3d(images, axis, cut_index, pad=1, rescale_mode='automatic', input_range=None):
    """ Puts a cut through the 3D kernel on the grid
    :param images: tensor of rank 5 with [batches, x, y, z, channels]
    :param axis: direction perpendicular to the cut, 0 for x, 1 for y, 2 for z
    :param cut_index: index where the cut is along the axis
    :param pad: number of black pixels around each filter (between them)
    :param rescale_mode: 'manual' or 'automatic
      Automatic rescale mode scales the images such that the they are in the range [0,255]
      Manual rescale mode maps input_range to [0,255] and thresholds everything outside the range
    :param input_range: input range used for manual rescaling
    :return:
    """
    image_cut = None
    if axis == 0:
        image_cut = images[:, cut_index, :, :, :]
    elif axis == 1:
        image_cut = images[:, :, cut_index, :, :]
    elif axis == 2:
        image_cut = images[:, :, :, cut_index, :]
    return put_kernels_on_grid(image_cut, pad, rescale_mode, input_range)