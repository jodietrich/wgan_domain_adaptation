# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers


# ---------------stand alone functions for discriminator and generator------------------------------
def only_conv_generator(z, training, residual=True, batch_normalization=False, hidden_layers=2, filters=16,
                        scope_name='generator', scope_reuse=False):
    # batch size 2: hidden_layers=2, filters=16
    # batch size 1: hidden_layers=3, filters=32
    # only residual connection from beginning to end possible
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()
        previous_layer = z
        for depth in range(1, hidden_layers + 1):
            if(batch_normalization):
                previous_layer = layers.conv3D_layer_bn(previous_layer, 'gconv%d' % depth, training, num_filters=filters,
                                                    activation=tf.nn.relu)
            else:
                previous_layer = layers.conv3D_layer(previous_layer, 'gconv%d' % depth, num_filters=32,
                                                     activation=tf.nn.relu)
        last_layer = layers.conv3D_layer(previous_layer, 'gconv%d_last' % (hidden_layers + 1), num_filters=1,
                                         kernel_size=(1, 1, 1), strides=(1, 1, 1), activation=tf.identity)
        if residual:
            return last_layer + z
        else:
            return last_layer


def pool_fc_discriminator_bs2(x, training, scope_name='discriminator', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(x, 'dconv1_1',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'dconv2_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'dconv3_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'dconv3_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'dconv4_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        conv4_2 = layers.conv3D_layer(conv4_1, 'dconv4_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool4 = layers.max_pool_layer3d(conv4_2)

        conv5_1 = layers.conv3D_layer(pool4, 'dconv5_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu)

        conv5_2 = layers.conv3D_layer(conv5_1, 'dconv5_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu)

        pool5 = layers.max_pool_layer3d(conv5_2)

        conv6_1 = layers.conv3D_layer(pool5, 'dconv6_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu)

        conv6_2 = layers.conv3D_layer(conv6_1, 'dconv6_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu)

        pool6 = layers.max_pool_layer3d(conv6_2)

        dense1 = layers.dense_layer(pool6, 'ddense1', hidden_units=256, activation=layers.leaky_relu)

        dense2 = layers.dense_layer(dense1, 'ddense2', hidden_units=1, activation=tf.identity)

        return dense2


def pool_fc_discriminator_bs2_bn(x, training, scope_name='discriminator', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(x, 'dconv1_1',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'dconv2_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'dconv3_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'dconv3_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'dconv4_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'dconv4_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool4 = layers.max_pool_layer3d(conv4_2)

        conv5_1 = layers.conv3D_layer_bn(pool4, 'dconv5_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu, training=training)

        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'dconv5_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu, training=training)

        pool5 = layers.max_pool_layer3d(conv5_2)

        conv6_1 = layers.conv3D_layer_bn(pool5, 'dconv6_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu, training=training)

        conv6_2 = layers.conv3D_layer_bn(conv6_1, 'dconv6_2',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                        activation=layers.leaky_relu, training=training)

        pool6 = layers.max_pool_layer3d(conv6_2)

        dense1 = layers.dense_layer(pool6, 'ddense1', hidden_units=256, activation=layers.leaky_relu)

        dense2 = layers.dense_layer(dense1, 'ddense2', hidden_units=1, activation=tf.identity)

        return dense2

def pool_fc_discriminator_bs1(x, training, scope_name='discriminator', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(x, 'dconv1_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'dconv2_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'dconv3_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'dconv3_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                    activation=layers.leaky_relu, training=training)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'dconv4_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        conv4_2 = layers.conv3D_layer(conv4_1, 'dconv4_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                    activation=layers.leaky_relu)

        pool4 = layers.max_pool_layer3d(conv4_2)

        conv5_1 = layers.conv3D_layer(pool4, 'dconv5_1',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                        activation=layers.leaky_relu)

        conv5_2 = layers.conv3D_layer(conv5_1, 'dconv5_2',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                        activation=layers.leaky_relu)

        pool5 = layers.max_pool_layer3d(conv5_2)

        conv6_1 = layers.conv3D_layer(pool5, 'dconv6_1',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                        activation=layers.leaky_relu)

        conv6_2 = layers.conv3D_layer(conv6_1, 'dconv6_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                        activation=layers.leaky_relu)

        pool6 = layers.max_pool_layer3d(conv6_2)

        dense1 = layers.dense_layer(pool6, 'ddense1', hidden_units=512, activation=layers.leaky_relu)

        dense2 = layers.dense_layer(dense1, 'ddense2', hidden_units=1, activation=tf.identity)

        return dense2


# Bousmalis Netzwerke
# can only be used with images in [-1, 1]
# TODO: include noise
def bousmalis_generator(x, training, batch_normalization, residual_blocks, nfilters, scope_name='generator', scope_reuse=False):
    kernel_size = (3, 3, 3)
    strides = (1, 1, 1)
    # define layer for the residual blocks
    if batch_normalization:
        conv_layer = lambda bottom, name, activation: layers.conv3D_layer_bn(bottom, name, training=training,
                                                                       kernel_size=kernel_size, num_filters=nfilters,
                                                                       strides=strides, activation=activation)
    else:
        conv_layer = lambda bottom, name, activation: layers.conv3D_layer(bottom, name, kernel_size=kernel_size,
                                                                    num_filters=nfilters, strides=strides,
                                                                    activation=activation)
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()
        previous_layer = layers.conv3D_layer(x, 'conv1', kernel_size=kernel_size, num_filters=nfilters, strides=strides,
                        activation=tf.nn.relu)

        # place residual blocks
        for block_num in range(1, 1 + residual_blocks):
            previous_layer = layers.residual_block_original(previous_layer, 'res_block_' + str(block_num), conv_layer,
                                                            activation=tf.nn.relu, nlayers=2)

        conv_out = layers.conv3D_layer(previous_layer, 'conv_out', kernel_size=kernel_size, num_filters=1, strides=strides,
                        activation=tf.nn.tanh)
        return conv_out


def bousmalis_discriminator(x, training, batch_normalization, middle_layers, nfilters, scope_name='discriminator', scope_reuse=False):
    # leaky relu has the same parameter as in the paper
    leaky_relu = lambda x: layers.leaky_relu(x, alpha=0.2)
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()
        if batch_normalization:
            previous_layer = layers.conv3D_layer_bn(x, 'convs1_1', kernel_size=(3,3,3), num_filters=nfilters, strides=(1,1,1),
                        activation=leaky_relu, training=training)
        else:
            previous_layer = layers.conv3D_layer(x, 'convs1_1', kernel_size=(3,3,3), num_filters=nfilters, strides=(1,1,1),
                        activation=leaky_relu)

        for layer in range(1, 1 + middle_layers):
            if batch_normalization:
                previous_layer = layers.conv3D_layer_bn(previous_layer, 'convs2_' + str(layer), kernel_size=(3,3,3), num_filters=nfilters*(2**layer), strides=(2,2,2),
                            activation=leaky_relu, training=training)
            else:
                previous_layer = layers.conv3D_layer(previous_layer, 'convs2_' + str(layer), kernel_size=(3,3,3), num_filters=nfilters*(2**layer), strides=(2,2,2),
                            activation=leaky_relu)

        dense_out = layers.dense_layer(previous_layer, 'dense_out', hidden_units=1, activation=tf.identity)

    return dense_out





    # classifiers from fieldstrength classifier, use too much memory to be used as discriminators

def jia_xi_net(images, training, nlabels, scope_name='classifier', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(images, 'conv1_1', num_filters=32)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'conv2_1', num_filters=64)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'conv3_1', num_filters=128)
        conv3_2 = layers.conv3D_layer(conv3_1, 'conv3_2', num_filters=128)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'conv4_1', num_filters=256)
        conv4_2 = layers.conv3D_layer(conv4_1, 'conv4_2', num_filters=256)

        pool4 = layers.max_pool_layer3d(conv4_2)

        dense1 = layers.dense_layer(pool4, 'dense1', hidden_units=512)
        dense2 = layers.dense_layer(dense1, 'dense2', hidden_units=nlabels, activation=tf.identity)

        return dense2


def jia_xi_net_bn(images, training, nlabels, scope_name='classifier', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training)

        pool4 = layers.max_pool_layer3d(conv4_2)

        dense1 = layers.dense_layer_bn(pool4, 'dense1', hidden_units=512, training=training)
        dense2 = layers.dense_layer_bn(dense1, 'dense2', hidden_units=nlabels, activation=tf.identity, training=training)

        return dense2


def jia_xi_net_multitask_ordinal(images, training, nlabels, n_age_thresholds=5, scope_name='classifier',
                                 scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(images, 'conv1_1', num_filters=32)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'conv2_1', num_filters=64)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'conv3_1', num_filters=128)
        conv3_2 = layers.conv3D_layer(conv3_1, 'conv3_2', num_filters=128)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'conv4_1', num_filters=256)
        conv4_2 = layers.conv3D_layer(conv4_1, 'conv4_2', num_filters=256)

        pool4 = layers.max_pool_layer3d(conv4_2)

        dense1 = layers.dense_layer(pool4, 'dense1', hidden_units=512)
        diagnosis = layers.dense_layer(dense1, 'dense2', hidden_units=nlabels, activation=tf.identity)

        dense_ages = layers.dense_layer(pool4, 'dense_ages', hidden_units=512)

        ages_logits = []
        for ii in range(n_age_thresholds):
            ages_logits.append(layers.dense_layer(dense_ages, 'age_%s' % str(ii),
                                                     hidden_units=2, activation=tf.identity))

        return diagnosis, ages_logits


def jia_xi_net_multitask_ordinal_bn(images, training, nlabels, n_age_thresholds=5, bn_momentum=0.99,
                                    scope_name='classifier', scope_reuse=False):
    # with tf.variable_scope(scope_name) as scope:
    #     if scope_reuse:
    #         scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training, bn_momentum=bn_momentum)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training, bn_momentum=bn_momentum)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training, bn_momentum=bn_momentum)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training, bn_momentum=bn_momentum)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training, bn_momentum=bn_momentum)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training, bn_momentum=bn_momentum)

        pool4 = layers.max_pool_layer3d(conv4_2)

        dense1 = layers.dense_layer_bn(pool4, 'dense1', hidden_units=512, training=training, bn_momentum=bn_momentum)
        diagnosis = layers.dense_layer_bn(dense1, 'dense2', hidden_units=nlabels, activation=tf.identity, training=training, bn_momentum=bn_momentum)

        dense_ages = layers.dense_layer_bn(pool4, 'dense_ages', hidden_units=512, training=training, bn_momentum=bn_momentum)

        ages_logits = []
        for ii in range(n_age_thresholds):
            ages_logits.append(layers.dense_layer_bn(dense_ages, 'age_%s' % str(ii),
                                                     hidden_units=2, activation=tf.identity, training=training, bn_momentum=bn_momentum))

        return diagnosis, ages_logits


def FCN_multitask_ordinal_bn(images, training, nlabels, n_age_thresholds=5, bn_momentum=0.99, scope_name='classifier',
                             scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()
        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training, bn_momentum=bn_momentum)

        pool1 = layers.max_pool_layer3d(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training, bn_momentum=bn_momentum)

        pool2 = layers.max_pool_layer3d(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training, bn_momentum=bn_momentum)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training, bn_momentum=bn_momentum)

        pool3 = layers.max_pool_layer3d(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training, bn_momentum=bn_momentum)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training, bn_momentum=bn_momentum)

        pool4 = layers.max_pool_layer3d(conv4_2)

        conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training, bn_momentum=bn_momentum)
        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training, bn_momentum=bn_momentum)

        convD_1 = layers.conv3D_layer_bn(conv5_2, 'convD_1', num_filters=256, training=training, bn_momentum=bn_momentum)
        convD_2 = layers.conv3D_layer_bn(convD_1,
                                         'convD_2',
                                         num_filters=nlabels,
                                         training=training,
                                         bn_momentum=bn_momentum,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        diag_logits = layers.reduce_avg_layer3D(convD_2, name='diagnosis_avg')

        convA_1 = layers.conv3D_layer_bn(pool4, 'convA_1', num_filters=256, training=training, bn_momentum=bn_momentum)

        ages_logits = []
        for ii in range(n_age_thresholds):

            age_activations = layers.conv3D_layer_bn(convA_1,
                                                     'convA_2_%d' % ii,
                                                     num_filters=2,
                                                     training=training,
                                                     bn_momentum=bn_momentum,
                                                     kernel_size=(1, 1, 1),
                                                     activation=tf.identity)

            ages_logits.append(layers.reduce_avg_layer3D(age_activations, name='age_avg_%d' % ii))


        return diag_logits, ages_logits

