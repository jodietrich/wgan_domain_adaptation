# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers



class DCGAN_FCN_bn:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()

            conv1_1 = layers.conv3D_layer_bn(x, 'dconv1_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            pool1 = layers.max_pool_layer3d(conv1_1)

            conv2_1 = layers.conv3D_layer_bn(pool1, 'dconv2_1',kernel_size=(3,3,3), num_filters=16, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            pool2 = layers.max_pool_layer3d(conv2_1)

            conv3_1 = layers.conv3D_layer_bn(pool2, 'dconv3_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            conv3_2 = layers.conv3D_layer_bn(conv3_1, 'dconv3_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            pool3 = layers.max_pool_layer3d(conv3_2)

            conv4_1 = layers.conv3D_layer_bn(pool3, 'dconv4_1',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            conv4_2 = layers.conv3D_layer_bn(conv4_1, 'dconv4_2',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                                        activation=layers.leaky_relu, training=training)

            pool4 = layers.max_pool_layer3d(conv4_2)

            conv5_1 = layers.conv3D_layer_bn(pool4, 'dconv5_1',kernel_size=(3,3,3), num_filters=64, strides=(1,1,1),
                            activation=layers.leaky_relu, training=training)

            conv5_2 = layers.conv3D_layer_bn(conv5_1, 'dconv5_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                            activation=layers.leaky_relu, training=training)

            pool5 = layers.max_pool_layer3d(conv5_2)

            dense1 = layers.dense_layer(pool5, 'ddense1', hidden_units=512, activation=layers.leaky_relu)

            dense2 = layers.dense_layer(dense1, 'ddense2', hidden_units=1, activation=tf.identity)

            return dense2

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer_bn(z, 'gconv1', num_filters=32, activation=tf.nn.relu,
                                         training=training)

            layer2 = layers.conv3D_layer_bn(layer1, 'gconv2', num_filters=32, activation=tf.nn.relu,
                                         training=training)

            layer3 = layers.conv3D_layer_bn(layer2, 'gconv3', num_filters=32, activation=tf.nn.relu,
                                         training=training)

            layer4 = layers.conv3D_layer(layer3, 'gconv4', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            return layer4



# not yet functional. Change the last layers of the discriminator and optionally change the discriminator
class FCN:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()

            conv1_1 = layers.conv3D_layer(x, 'dconv1_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu)

            pool1 = layers.max_pool_layer3d(conv1_1)

            conv2_1 = layers.conv3D_layer(pool1, 'dconv2_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu)

            conv2_2 = layers.conv3D_layer(conv2_1, 'dconv2_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu)

            pool2 = layers.max_pool_layer3d(conv2_2)

            conv3_1 = layers.conv3D_layer(pool2, 'dconv3_1',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu)

            conv3_2 = layers.conv3D_layer(conv3_1, 'dconv3_2',kernel_size=(3,3,3), num_filters=32, strides=(1,1,1),
                                        activation=layers.leaky_relu)

            conv3_3 = layers.conv3D_layer(conv3_2, 'dconv3_3', num_filters=2, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.identity)

            return conv3_3

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer_bn(z, 'glayer1', num_filters=32, activation=tf.nn.relu,
                                         training=training)

            layer2 = layers.conv3D_layer_bn(layer1, 'glayer2', num_filters=32, activation=tf.nn.relu,
                                         training=training)

            layer3 = layers.conv3D_layer(layer2, 'glayer3', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.sigmoid)

            return layer3


