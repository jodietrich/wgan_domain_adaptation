# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import losses
from math import sqrt

def gan_loss(logits_real, logits_fake, w_reg_gen_l1, w_reg_disc_l1, w_reg_gen_l2, w_reg_disc_l2):

    disc_loss = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)
    gen_loss = tf.reduce_mean(logits_fake)

    tf.summary.scalar('discriminator_loss', disc_loss)
    tf.summary.scalar('generator_loss', gen_loss)

    gen_weights = [v for v in tf.get_collection('weight_variables') if v.name.startswith('generator')]
    disc_weights = [v for v in tf.get_collection('weight_variables') if v.name.startswith('discriminator')]

    with tf.variable_scope('weights_norm') as scope:

        # Discriminator regularisation
        l1_regularizer_disc = tf.contrib.layers.l1_regularizer(scale=w_reg_disc_l1, scope=None)
        l2_regularizer_disc = tf.contrib.layers.l2_regularizer(scale=w_reg_disc_l2, scope=None)
        reg_disc_l1 = tf.contrib.layers.apply_regularization(l1_regularizer_disc, disc_weights)
        reg_disc_l2 = tf.contrib.layers.apply_regularization(l2_regularizer_disc, disc_weights)
        reg_disc = reg_disc_l1 + reg_disc_l2

        # Generator regularisation
        l1_regularizer_gen = tf.contrib.layers.l1_regularizer(scale=w_reg_gen_l1, scope=None)
        l2_regularizer_gen = tf.contrib.layers.l2_regularizer(scale=w_reg_gen_l2, scope=None)
        reg_gen_l1 = tf.contrib.layers.apply_regularization(l1_regularizer_gen, gen_weights)
        reg_gen_l2 = tf.contrib.layers.apply_regularization(l2_regularizer_gen, gen_weights)
        reg_gen = reg_gen_l1 + reg_gen_l2

        reg_all = reg_disc + reg_gen


    print('== REGULARISATION SUMMARY ==')
    print('Discriminator Regularisation:')
    print(' - L1: %f' % w_reg_disc_l1)
    print(' - L2: %f' % w_reg_disc_l2)
    for v in disc_weights:
        print(v.name)

    print('Generator Regularisation:')
    print(' - L1: %f' % w_reg_gen_l1)
    print(' - L2: %f' % w_reg_gen_l2)
    for v in gen_weights:
        print(v.name)

    total_disc_loss = disc_loss + reg_all
    total_gen_loss = gen_loss + reg_all

    return total_disc_loss, total_gen_loss, disc_loss, gen_loss


def train_step(loss_val, var_list, optimizer_handle, learning_rate):

    # The with statement is needed to make sure batch norm properly performs its updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer_handle == tf.train.AdamOptimizer:
            optimizer = optimizer_handle(learning_rate, beta1=0.5, beta2=0.9)
        else:
            optimizer = optimizer_handle(learning_rate)

        train_op = optimizer.minimize(loss_val, var_list=var_list)

    return train_op

    # for future: add gradient summary
    # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    # # for grad, var in grads:
    # #     utils.add_gradient_summary(grad, var)
    # return optimizer.apply_gradients(grads)

def clip_op():

    train_variables = tf.trainable_variables()
    d_clip_op = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in train_variables if v.name.startswith("discriminator") and '_bn' not in v.name]

    print('== CLIPPED VARIABLES SUMMARY ==')
    [print(v.name) for v in train_variables if v.name.startswith("discriminator") and '_bn' not in v.name]

    return d_clip_op

def training_ops(logits_real,
                 logits_fake,
                 optimizer_handle,
                 learning_rate,
                 w_reg_gen_l1=2.5e-5,
                 w_reg_disc_l1=2.5e-5,
                 w_reg_gen_l2=0.0,
                 w_reg_disc_l2=0.0,
                 d_hat=None,
                 x_hat=None,
                 scale=10.0):

    train_variables = tf.trainable_variables()

    generator_variables = [v for v in train_variables if v.name.startswith("generator")]
    discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
    # discriminator_variables = [v for v in train_variables]

    print('== TRAINED VARIABLES SUMMARIES ==')
    print(' - Generator variables:')
    for v in generator_variables:
        print(v.name)
    print(' - Discriminator variables:')
    for v in discriminator_variables:
        print(v.name)


    discriminator_loss, gen_loss, discriminator_loss_no_reg, gen_loss_no_reg = gan_loss(logits_real,
                                                                                        logits_fake,
                                                                                        w_reg_gen_l1,
                                                                                        w_reg_disc_l1,
                                                                                        w_reg_gen_l2,
                                                                                        w_reg_disc_l2)

    if d_hat is not None and x_hat is not None:

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        discriminator_loss = discriminator_loss + ddx


    generator_train_op = train_step(gen_loss, generator_variables, optimizer_handle, learning_rate)
    discriminator_train_op = train_step(discriminator_loss, discriminator_variables, optimizer_handle, learning_rate)

    return discriminator_train_op, generator_train_op, discriminator_loss, gen_loss, discriminator_loss_no_reg, gen_loss_no_reg





