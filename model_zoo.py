# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers


# preactivation residual unit as generator to have identity function as starting point for the generator
class ResNet_gen_bs2_bn:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
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

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            bn = layers.batch_normalisation_layer(z, 'gbn', training=training)

            activation = layers.activation_layer(bn, 'gactivation')

            conv1 = layers.conv3D_layer_bn(activation, 'gconv1', num_filters=16, activation=tf.nn.relu,
                                         training=training)

            conv2 = layers.conv3D_layer_bn(conv1, 'gconv2', num_filters=16, activation=tf.nn.relu,
                                         training=training)

            conv3 = layers.conv3D_layer(conv2, 'gconv3', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            residual_out = z + conv3

            return residual_out




class Std_CNN_bs2_bn:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
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

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer_bn(z, 'gconv1', num_filters=16, activation=tf.nn.relu,
                                         training=training)

            layer2 = layers.conv3D_layer_bn(layer1, 'gconv2', num_filters=16, activation=tf.nn.relu,
                                         training=training)

            layer3 = layers.conv3D_layer(layer2, 'gconv3', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            return layer3



class Std_CNN_bs2:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
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

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer(z, 'gconv1', num_filters=16, activation=tf.nn.relu)

            layer2 = layers.conv3D_layer(layer1, 'gconv2', num_filters=16, activation=tf.nn.relu)

            layer3 = layers.conv3D_layer(layer2, 'gconv3', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            return layer3


# bigger architecture that only fits into memory when using batch size 1
class Std_CNN_bs1:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
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

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer(z, 'gconv1', num_filters=32, activation=tf.nn.relu)

            layer2 = layers.conv3D_layer(layer1, 'gconv2', num_filters=32, activation=tf.nn.relu)

            layer3 = layers.conv3D_layer(layer2, 'gconv3', num_filters=32, activation=tf.nn.relu)

            layer4 = layers.conv3D_layer(layer3, 'gconv4', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            return layer4



