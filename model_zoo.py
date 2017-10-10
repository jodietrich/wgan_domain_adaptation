# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers

# class DCGAN:
#
#     @staticmethod
#     def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
#
#         with tf.variable_scope(scope_name) as scope:
#
#             if scope_reuse:
#                 scope.reuse_variables()
#
#             layer1 = layers.conv2D_layer(x, 'dlayer1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, activation=layers.leaky_relu, weight_init='simple')
#             layer2 = layers.conv2D_layer(layer1, 'dlayer2', kernel_size=(4,4), strides=(2,2), num_filters=128, activation=layers.leaky_relu, weight_init='simple')
#             layer3 = layers.dense_layer(layer2, 'dlayer3', hidden_units=1024, activation=layers.leaky_relu, weight_init='simple')
#             layer4 = layers.dense_layer(layer3, 'dlayer4', hidden_units=1, activation=tf.identity, weight_init='simple')
#
#             return layer4
#
#
#     @staticmethod
#     def generator(z, training, scope_name='generator'):
#
#         with tf.variable_scope(scope_name):
#
#             bs = tf.shape(z)[0]
#
#             layer1 = layers.dense_layer_bn(z, 'glayer1', hidden_units=1024, activation=tf.nn.relu, weight_init='simple', training=training)
#             layer2 = layers.dense_layer_bn(layer1, 'glayer2', hidden_units=7*7*128, activation=tf.nn.relu, weight_init='simple', training=training)
#             layer2 = tf.reshape(layer2, tf.stack([bs, 7, 7, 128]))
#
#             layer3 = layers.deconv2D_layer_bn(layer2, 'glayer3', kernel_size=(4,4), strides=(2,2), num_filters=64, activation=tf.nn.relu, weight_init='simple', training=training)
#             layer4 = layers.deconv2D_layer(layer3, 'glayer4', kernel_size=(4,4), strides=(2,2), num_filters=1, activation=tf.sigmoid, weight_init='simple')
#
#             return layer4
#
#
# class DCGAN_FCN:
#     @staticmethod
#     def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
#         with tf.variable_scope(scope_name) as scope:
#             if scope_reuse:
#                 scope.reuse_variables()
#
#             layer1 = layers.conv2D_layer(x, 'dlayer1', kernel_size=(4, 4), strides=(2, 2), num_filters=64,
#                                          activation=layers.leaky_relu, weight_init='simple')
#             layer2 = layers.conv2D_layer(layer1, 'dlayer2', kernel_size=(4, 4), strides=(2, 2), num_filters=128,
#                                          activation=layers.leaky_relu, weight_init='simple')
#             layer3 = layers.dense_layer(layer2, 'dlayer3', hidden_units=1024, activation=layers.leaky_relu,
#                                         weight_init='simple')
#             layer4 = layers.dense_layer(layer3, 'dlayer4', hidden_units=1, activation=tf.identity,
#                                         weight_init='simple')
#
#             return layer4
#
#     @staticmethod
#     def generator(z, training, scope_name='generator'):
#         with tf.variable_scope(scope_name):
#
#             layer1 = layers.conv2D_layer(z, 'glayer1', num_filters=64, activation=tf.nn.relu,
#                                            weight_init='simple', training=training)
#
#             layer2 = layers.max_pool_layer2d(layer1)
#
#             layer3 = layers.conv2D_layer(layer2, 'glayer3', num_filters=128, activation=tf.nn.relu,
#                                            weight_init='simple', training=training)
#
#             layer4 = layers.deconv2D_layer(layer3, 'glayer4', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='simple')
#
#             layer5 = layers.conv2D_layer(layer4, 'glayer5', num_filters=128, activation=tf.nn.relu,
#                                          weight_init='simple')
#
#             layer6 = layers.conv2D_layer(layer5, 'glayer6', num_filters=1, kernel_size=(1,1), strides=(1,1), activation=tf.sigmoid)
#
#
#             return layer6


class DCGAN_FCN_bn:
    @staticmethod
    def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()

            conv1 = layers.conv3D_layer(x, 'dconv1',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                        activation=layers.leaky_relu, weight_init='simple')

            pool1 = layers.max_pool_layer3d(conv1)

            conv2 = layers.conv3D_layer(pool1, 'dconv2',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                        activation=layers.leaky_relu, weight_init='simple')

            pool2 = layers.max_pool_layer3d(conv2)

            conv3 = layers.conv3D_layer(pool2, 'dconv3',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                        activation=layers.leaky_relu, weight_init='simple')

            pool3 = layers.max_pool_layer3d(conv3)

            conv4 = layers.conv3D_layer(pool3, 'dconv4',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                                        activation=layers.leaky_relu, weight_init='simple')

            pool4 = layers.max_pool_layer3d(conv4)

            conv5 = layers.conv3D_layer(pool4, 'dconv5',kernel_size=(3,3,3), num_filters=8, strides=(1,1,1),
                            activation=layers.leaky_relu, weight_init='simple')

            pool5 = layers.max_pool_layer3d(conv5)

            dense1 = layers.dense_layer(pool5, 'ddense1', hidden_units=512, activation=layers.leaky_relu,
                                        weight_init='simple')

            dense2 = layers.dense_layer(dense1, 'ddense2', hidden_units=1, activation=tf.identity,
                                        weight_init='simple')

            return dense2

    @staticmethod
    def generator(z, training, scope_name='generator'):
        with tf.variable_scope(scope_name):
            layer1 = layers.conv3D_layer_bn(z, 'glayer1', num_filters=64, activation=tf.nn.relu,
                                         weight_init='simple', training=training)

            layer2 = layers.conv3D_layer_bn(layer1, 'glayer2', num_filters=128, activation=tf.nn.relu,
                                         weight_init='simple', training=training)

            layer3 = layers.conv3D_layer_bn(layer2, 'glayer3', num_filters=128, activation=tf.nn.relu,
                                         weight_init='simple', training=training)

            layer4 = layers.conv3D_layer(layer3, 'glayer4', num_filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                         activation=tf.sigmoid)

            return layer4

#
# class UNET:
#
#     @staticmethod
#     def discriminator(x, training, scope_name='discriminator', scope_reuse=False):
#
#         with tf.variable_scope(scope_name) as scope:
#
#             if scope_reuse:
#                 scope.reuse_variables()
#
#             bs = tf.shape(x)[0]
#             x = tf.reshape(x, [bs, 28, 28, 1])
#
#             layer1 = layers.conv2D_layer(x, 'dlayer1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, activation=layers.leaky_relu, weight_init='simple')
#             layer2 = layers.conv2D_layer(layer1, 'dlayer2', kernel_size=(4,4), strides=(2,2), num_filters=128, activation=layers.leaky_relu, weight_init='simple')
#             layer3 = layers.dense_layer(layer2, 'dlayer3', hidden_units=1024, activation=layers.leaky_relu, weight_init='simple')
#             layer4 = layers.dense_layer(layer3, 'dlayer4', hidden_units=1, activation=tf.identity, weight_init='simple')
#
#             return layer4
#
#
#     @staticmethod
#     def generator(z, training, scope_name='generator'):
#
#         with tf.variable_scope(scope_name):
#             conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
#             conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
#
#             pool1 = layers.max_pool_layer2d(conv1_2)
#
#             conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
#             conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
#
#             pool2 = layers.max_pool_layer2d(conv2_2)
#
#             conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
#             conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
#
#             upconv4 = layers.deconv2D_layer_bn(conv3_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2),
#                                                num_filters=512, training=training)
#             concat4 = tf.concat([conv3_2, upconv4], axis=3, name='concat4')
#
#             conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=128, training=training)
#             conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training)
#
#             upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2),
#                                                num_filters=256, training=training)
#             concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
#
#             conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=64, training=training)
#             conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training)
#
#             pred = layers.conv2D_layer_bn(conv7_2, 'pred', num_filters=1, kernel_size=(1, 1), activation=tf.sigmoid,
#                                           training=training)
#
#             return pred
#
#
