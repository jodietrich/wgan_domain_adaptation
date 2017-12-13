# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import math
import numpy as np
import logging

from tfwrapper import utils

from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)

STANDARD_NONLINEARITY = leaky_relu  # was tf.nn.relu

def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def max_pool_layer3d(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def reduce_avg_layer3D(x, name=None):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    op = tf.reduce_mean(x, axis=(1,2,3), keep_dims=False, name=name)
    tf.summary.histogram(op.op.name + '/activations', op)

    return op


def reduce_avg_layer2D(x, name=None):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    op = tf.reduce_mean(x, axis=(1,2), keep_dims=False, name=name)
    tf.summary.histogram(op.op.name + '/activations', op)

    return op


def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()[1:]
    # output_size = tf.shape(inputs[0])[1:]
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()[1:]
        # larger_size = tf.shape(inputs[ii])

        # Don't subtract over batch_size because it may be None
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 4:  # 3D images
        # if output_size.shape[0] == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[0], start_crop[1], start_crop[2], 0),
                                     (-1, output_size[0], output_size[1], output_size[2], -1))
        elif len(output_size) == 3:  # 2D images
        # elif output_size.shape[0] == 4:
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[0], start_crop[1], 0),
                                     (-1, output_size[0], output_size[1], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)


def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)


def dropout_layer(bottom, name, training, keep_prob=0.5):
    '''
    Performs dropout on the activations of an input
    '''

    keep_prob_pl = tf.cond(training,
                           lambda: tf.constant(keep_prob, dtype=bottom.dtype),
                           lambda: tf.constant(1.0, dtype=bottom.dtype))

    # The tf.nn.dropout function takes care of all the scaling
    # (https://www.tensorflow.org/get_started/mnist/pros)
    return tf.nn.dropout(bottom, keep_prob=keep_prob_pl, name=name)


def parametrize_variable(global_step, y_min, y_max, x_min, x_max):

    # if x < x_min:
    #     return y_min
    # elif x > x_max:
    #     return y_max
    # else:
    #     return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min

    x = tf.to_float(global_step)

    def f1(): return tf.constant(y_min)
    def f2(): return tf.constant(y_max)
    def f3(): return ((x - x_min) * (y_max - y_min) / (x_max - x_min)) + y_min

    y =  tf.case({tf.less(x, x_min): f1,
                  tf.greater(x, x_max): f2},
                  default=f3,
                  exclusive=True)

    return y


def layer_norm(x,
               gamma=None,
               beta=None,
               axes=(1, 2, 3),
               eps=1e-3,
               scope="ln",
               name="ln_out",
               return_mean=False):


    """Applies layer normalization.
    Collect mean and variances on x except the first dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, ...].
        axes: Axes to collect statistics.
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Layer-normalized variable.
        mean: Mean used for normalization (optional).
    """

    # raise NotImplementedError

    with tf.variable_scope(scope):
        x_shape = [x.get_shape()[-1]]
        mean, var = tf.nn.moments(x, axes, name='moments', keep_dims=True)
        normed = (x - mean) / tf.sqrt(eps + var)
        if gamma is not None:
          normed *= gamma
        if beta is not None:
          normed += beta
        normed = tf.identity(normed, name=name)
    if return_mean:
        return normed, mean
    else:
        return normed


def div_norm_2d(x,
                sum_window,
                sup_window,
                gamma=None,
                beta=None,
                eps=1.0,
                scope="dn",
                name="dn_out",
                return_mean=False):

    """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, H, W, C].
        sum_window: Summation window size, [H_sum, W_sum].
        sup_window: Suppression window size, [H_sup, W_sup].
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Divisive-normalized variable.
        mean: Mean used for normalization (optional).
    """
    with tf.variable_scope(scope):
        w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
        w_sup = tf.ones(sup_window + [1, 1]) / np.prod(np.array(sum_window))
        x_mean = tf.reduce_mean(x, [3], keep_dims=True)
        x_mean = tf.nn.conv2d(x_mean, w_sum, strides=[1, 1, 1, 1], padding='SAME')
        normed = x - x_mean
        x2 = tf.square(normed)
        x2_mean = tf.reduce_mean(x2, [3], keep_dims=True)
        x2_mean = tf.nn.conv2d(x2_mean, w_sup, strides=[1, 1, 1, 1], padding='SAME')
        denom = tf.sqrt(x2_mean + eps)
        normed = normed / denom
        if gamma is not None:
            normed *= gamma
        if beta is not None:
            normed += beta
        normed = tf.identity(normed, name=name)
    if return_mean:
        return normed, x_mean
    else:
        return normed


def div_norm_3d(x,
                sum_window,
                sup_window,
                gamma=None,
                beta=None,
                eps=1.0,
                scope="dn",
                name="dn_out",
                return_mean=False):

    """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, H, W, D, C].
        sum_window: Summation window size, [H_sum, W_sum, D_sum].
        sup_window: Suppression window size, [H_sup, W_sup, D_sup].
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Divisive-normalized variable.
        mean: Mean used for normalization (optional).
    """
    with tf.variable_scope(scope):

        bottom_num_filters = x.get_shape().as_list()[-1]

        w_sum = tf.ones(sum_window + [bottom_num_filters, 1]) / np.prod(np.array(sum_window))
        w_sup = tf.ones(sup_window + [bottom_num_filters, 1]) / np.prod(np.array(sum_window))
        x_mean = tf.reduce_mean(x, [3], keep_dims=True)
        x_mean = tf.nn.conv3d(x_mean, w_sum, strides=[1, 1, 1, 1, 1], padding='SAME')
        normed = x - x_mean
        x2 = tf.square(normed)
        x2_mean = tf.reduce_mean(x2, [3], keep_dims=True)
        x2_mean = tf.nn.conv3d(x2_mean, w_sup, strides=[1, 1, 1, 1, 1], padding='SAME')
        denom = tf.sqrt(x2_mean + eps)
        normed = normed / denom
        if gamma is not None:
            normed *= gamma
        if beta is not None:
            normed += beta
        normed = tf.identity(normed, name=name)
    if return_mean:
        return normed, x_mean
    else:
        return normed



def batch_renormalisation_layer(bottom, name, training, moving_average_decay=0.99):
    '''
    Alternative wrapper for tensorflows own batch normalisation function. I had some issues with 3D convolutions
    when using this. The version below works with 3D convs as well.
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    rmin = 1.0  # tf.constant(1.0, dtype=tf.float32)
    rmax = 3.0  #tf.constant(3.0, dtype=tf.float32)

    dmin = 0.0  #tf.constant(0.0, dtype=tf.float32)
    dmax = 5.0  #tf.constant(5.0, dtype=tf.float32)

    # values exactly /10 from paper because training goes faster for us
    x_min_r = 5000.0 / 4
    x_max_r = 40000.0 / 4

    x_min_d = 5000.0 / 4
    x_max_d = 25000.0 / 4

    global_step = tf.train.get_or_create_global_step()

    clip_r = parametrize_variable(global_step, rmin, rmax, x_min_r, x_max_r)
    clip_d = parametrize_variable(global_step, dmin, dmax, x_min_d, x_max_d)

    tf.summary.scalar('rmax_clip', clip_r)
    tf.summary.scalar('dmax_clip', clip_d)


    h_bn = tf.contrib.layers.batch_norm(inputs=bottom,
                                        renorm_decay=moving_average_decay,
                                        epsilon=1e-3,
                                        is_training=training,
                                        scope=name,
                                        center=True,
                                        scale=True,
                                        renorm=True,
                                        renorm_clipping={'rmax': clip_r, 'dmax': clip_d})

    return h_bn

#
def batch_normalisation_layer(bottom, name, training, moving_average_decay=0.99):
    '''
    Alternative wrapper for tensorflows own batch normalisation function. I had some issues with 3D convolutions
    when using this. The version below works with 3D convs as well.
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    h_bn = tf.contrib.layers.batch_norm(inputs=bottom,
                                        decay=moving_average_decay,
                                        epsilon=1e-3,
                                        is_training=training,
                                        scope=name,
                                        center=True,
                                        scale=True)

    return h_bn

#
# def batch_normalisation_layer(bottom, name, training, moving_average_decay=0.99, epsilon=1e-3):
#     '''
#     Batch normalisation layer (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
#     :param bottom: Input layer (should be before activation)
#     :param name: A name for the computational graph
#     :param training: A tf.bool specifying if the layer is executed at training or testing time
#     :return: Batch normalised activation
#     '''
#
#     with tf.variable_scope(name):
#
#         n_out = bottom.get_shape().as_list()[-1]
#         tensor_dim = len(bottom.get_shape().as_list())
#
#         if tensor_dim == 2:
#             # must be a dense layer
#             moments_over_axes = [0]
#         elif tensor_dim == 4:
#             # must be a 2D conv layer
#             moments_over_axes = [0, 1, 2]
#         elif tensor_dim == 5:
#             # must be a 3D conv layer
#             moments_over_axes = [0, 1, 2, 3]
#         else:
#             # is not likely to be something reasonable
#             raise ValueError('Tensor dim %d is not supported by this batch_norm layer' % tensor_dim)
#
#         init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
#         init_gamma = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
#         beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
#                                trainable=True)
#         gamma = tf.get_variable(name='gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
#                                 trainable=True)
#
#         batch_mean, batch_var = tf.nn.moments(bottom, moments_over_axes, name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(training, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normalised = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, epsilon)
#
#         return normalised

### FEED_FORWARD LAYERS ##############################################################################33

def residual_unit3D(bottom, name, num_filters, down_sample=False, projection=False, do_batch_norm=True, activation=STANDARD_NONLINEARITY, **kwargs):

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    assert (bottom_num_filters == num_filters) or (bottom_num_filters*2 == num_filters), \
        'Number of filters must remain constant, or be increased by a ' \
        'factor of 2. In filters: %d, Out filters: %d' % (bottom_num_filters, num_filters)

    increase_nfilters = True if bottom_num_filters < num_filters else False

    add_bias = not do_batch_norm  # don't add bias if doing batch norm

    if do_batch_norm:
        assert 'training' in kwargs, 'If batch norm is enabled, the "training" argument must be given'
        training = kwargs['training']

    if down_sample:
        first_strides = (2,2,2)
    else:
        first_strides = (1,1,1)

    with tf.variable_scope(name):

        conv1 = conv3D_layer(bottom, 'conv1', num_filters=num_filters, strides=first_strides, activation=tf.identity, add_bias=add_bias)
        if do_batch_norm:
            conv1 = batch_normalisation_layer(conv1, 'bn1', training=training)
        conv1 = activation(conv1)

        conv2 = conv3D_layer(conv1, 'conv2', num_filters=num_filters, strides=(1,1,1), activation=tf.identity, add_bias=add_bias)
        if do_batch_norm:
            conv2 = batch_normalisation_layer(conv2, 'bn2', training=training)
        # conv2 = activation(conv2)

        if increase_nfilters:
            if projection:
                projection = conv3D_layer(bottom, 'projection', num_filters=num_filters, strides=first_strides, kernel_size=(1,1,1), activation=tf.identity, add_bias=add_bias)
                if do_batch_norm:
                    projection = batch_normalisation_layer(projection, 'bn_projection', training=training)
                skip = activation(projection)
            else:
                pad_size = (num_filters - bottom_num_filters) // 2
                identity = tf.pad(bottom, paddings=[[0, 0],[0,0],[0,0],[0,0], [pad_size, pad_size]])
                if down_sample:
                    identity = identity[:,::2,::2,::2,:]
                skip = identity
        else:
            skip = bottom

        block = tf.add(skip, conv2, name='add')
        block = activation(block)

        return block


def identity_residual_unit3D(bottom, name, num_filters, down_sample=False, projection=False, do_batch_norm=True, activation=STANDARD_NONLINEARITY, **kwargs):

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    assert (bottom_num_filters == num_filters) or (bottom_num_filters*2 == num_filters), \
        'Number of filters must remain constant, or be increased by a ' \
        'factor of 2. In filters: %d, Out filters: %d' % (bottom_num_filters, num_filters)

    increase_nfilters = True if bottom_num_filters < num_filters else False

    add_bias = not do_batch_norm  # don't add bias if doing batch norm

    if do_batch_norm:
        assert 'training' in kwargs, 'If batch norm is enabled, the "training" argument must be given'
        training = kwargs['training']

    if down_sample:
        first_strides = (2,2,2)
    else:
        first_strides = (1,1,1)

    with tf.variable_scope(name):

        if do_batch_norm:
            op1 = batch_normalisation_layer(bottom, 'bn1', training=training)
        else:
            op1 = bottom
        op1 = activation(op1)
        op1 = conv3D_layer(op1, 'conv1', num_filters=num_filters, strides=first_strides, activation=tf.identity, add_bias=add_bias)

        if do_batch_norm:
            op2 = batch_normalisation_layer(op1, 'bn2', training=training)
        else:
            op2 = op1
        op2 = activation(op2)
        op2 = conv3D_layer(op2, 'conv2', num_filters=num_filters, strides=first_strides, activation=tf.identity, add_bias=add_bias)


        if increase_nfilters:
            if projection:
                projection = conv3D_layer(bottom, 'projection', num_filters=num_filters, strides=first_strides, kernel_size=(1,1,1), activation=tf.identity, add_bias=add_bias)
                if do_batch_norm:
                    projection = batch_normalisation_layer(projection, 'bn_projection', training=training)
                skip = activation(projection)
            else:
                pad_size = (num_filters - bottom_num_filters) // 2
                identity = tf.pad(bottom, paddings=[[0, 0],[0,0],[0,0],[0,0], [pad_size, pad_size]])
                if down_sample:
                    identity = identity[:,::2,::2,::2,:]
                skip = identity
        else:
            skip = bottom

        block = tf.add(skip, op2, name='add')

        return block


def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=STANDARD_NONLINEARITY,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 2D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=STANDARD_NONLINEARITY,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True,
                 init_weights=None,
                 init_biases=None):

    '''
    Standard 3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]
    #bottom_num_filters = tf.shape(bottom)[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True, init_weights=init_weights)
        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b', init_biases=init_biases)
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(4,4),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=STANDARD_NONLINEARITY,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()
    if output_shape is None:
        batch_size = tf.shape(bottom)[0]
        output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]
    bias_shape = [num_filters]
    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=STANDARD_NONLINEARITY,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        batch_size = tf.shape(bottom)[0]
        output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        # op = tf.reshape(op, output_shape)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv2D_dilated_layer(bottom,
                         name,
                         kernel_size=(3,3),
                         num_filters=32,
                         rate=2,
                         activation=STANDARD_NONLINEARITY,
                         padding="SAME",
                         weight_init='he_normal',
                         add_bias=True):

    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=STANDARD_NONLINEARITY,
                weight_init='he_normal',
                add_bias=True):

    '''
    Dense a.k.a. fully connected layer
    '''

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.matmul(bottom_flat, weights)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    bn_momentum=0.99):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training, moving_average_decay=bn_momentum)

    act = activation(conv_bn)

    return act


def conv3D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    bn_momentum=0.99,
                    init_weights=None,
                    init_biases=None):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False,
                        init_weights=init_weights,
                        init_biases=init_biases
                        )

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training, moving_average_decay=bn_momentum)

    act = activation(conv_bn)

    return act


def conv3D_layer_ln(bottom,
                    name,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    init_weights=None,
                    init_biases=None):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False,
                        init_weights=init_weights,
                        init_biases=init_biases
                        )

    conv_ln = layer_norm(conv, scope=name, name='ln', axes=(1,2,3,4))

    act = activation(conv_ln)

    return act


def conv3D_layer_lrn(bottom,
                    name,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    init_weights=None,
                    init_biases=None):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False,
                        init_weights=init_weights,
                        init_biases=init_biases
                        )

    conv_ln = div_norm_3d(conv, sum_window=[3,3,3], sup_window=[3,3,3], scope=name, name='ln')

    act = activation(conv_ln)

    return act


def conv3D_layer_bren(bottom,
                    name,
                    training,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    bn_momentum=0.99):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False,
                        )

    conv_bn = batch_renormalisation_layer(conv, name + '_bn', training, moving_average_decay=bn_momentum)

    act = activation(conv_bn)

    return act


def deconv2D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=STANDARD_NONLINEARITY,
                      padding="SAME",
                      weight_init='he_normal',
                      bn_momentum=0.99):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training, moving_average_decay=bn_momentum)

    act = activation(deco_bn)

    return act


def deconv3D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=STANDARD_NONLINEARITY,
                      padding="SAME",
                      weight_init='he_normal',
                      bn_momentum=0.99):

    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training, moving_average_decay=bn_momentum)

    act = activation(deco_bn)

    return act


def conv2D_dilated_layer_bn(bottom,
                           name,
                           training,
                           kernel_size=(3,3),
                           num_filters=32,
                           rate=1,
                           activation=STANDARD_NONLINEARITY,
                           padding="SAME",
                           weight_init='he_normal',
                           bn_momentum=0.99):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=tf.identity,
                                padding=padding,
                                weight_init=weight_init,
                                add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training, moving_average_decay=bn_momentum)

    act = activation(conv_bn)

    return act



def dense_layer_bn(bottom,
                   name,
                   training,
                   hidden_units=512,
                   activation=STANDARD_NONLINEARITY,
                   weight_init='he_normal',
                   bn_momentum=0.99):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=tf.identity,
                         weight_init=weight_init,
                         add_bias=False)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training, moving_average_decay=bn_momentum)
    act = activation(batchnorm)

    return act

### VARIABLE INITIALISERS ####################################################################################

def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):

    if 'init_weights' in kwargs and kwargs['init_weights'] is not None:
        type = 'pretrained'
        logging.info('Using pretrained weights for layer: %s' % name)

    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'pretrained':
        initial = kwargs.get('init_weights')
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.add_to_collection('weight_variables', weight)

    return weight



def get_bias_variable(shape, name=None, init_value=0.0, **kwargs):

    if 'init_biases' in kwargs and kwargs['init_biases'] is not None:
        initial = kwargs['init_biases']
        logging.info('Using pretrained weights for layer: %s' % name)
    else:
        initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)



def _upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights

def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name, weights)
    if biases:
        tf.summary.histogram(biases.name, biases)
    tf.summary.histogram(op.op.name + '/activations', op)