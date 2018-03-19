# Authors:
# Jonathan Dietrich

# training operation for joint training of translator and classifier


import tensorflow as tf
from tfwrapper import losses
from math import sqrt

import gan_model

def training_ops(logits_real,
                 logits_fake,
                 classifier_loss,
                 optimizer_handle,
                 learning_rate_gan,
                 learning_rate_clf,
                 l1_img_dist,
                 gan_loss_weight = 1,
                 task_loss_weight = 1e5,
                 w_reg_img_dist_l1=0.0,
                 w_reg_gen_l1=2.5e-5,
                 w_reg_disc_l1=2.5e-5,
                 w_reg_gen_l2=0.0,
                 w_reg_disc_l2=0.0,
                 d_hat=None,
                 x_hat=None,
                 scale=10.0):

    inner_dict = {'nr': None, 'reg': None}
    losses = {network: inner_dict.copy() for network in ['disc', 'gen']}
    # nr means no regularization, meaning the loss without the regularization term, reg is with regularization
    [losses['disc']['reg'], losses['gen']['reg'], losses['disc']['nr'], losses['gen']['nr']] = gan_model.gan_loss(logits_real,
                                                                                        logits_fake,
                                                                                        l1_img_dist,
                                                                                        w_reg_img_dist_l1,
                                                                                        w_reg_gen_l1,
                                                                                        w_reg_disc_l1,
                                                                                        w_reg_gen_l2,
                                                                                        w_reg_disc_l2)

    # improved training
    if d_hat is not None and x_hat is not None:
        ddx = gan_model.improved_training_regularization(d_hat, x_hat, scale)
        losses['disc']['reg'] = losses['disc']['reg'] + ddx

    train_variables = tf.trainable_variables()
    generator_variables = [v for v in train_variables if v.name.startswith("generator")]
    discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
    classifier_variables = [v for v in train_variables if v.name.startswith("classifier")]

    # redefine the losses for the joint training (include classifier and interaction weights)
    losses['disc']['joint'] = gan_loss_weight*losses['disc']['reg']
    losses['gen']['joint'] = gan_loss_weight*losses['gen']['reg'] + task_loss_weight*classifier_loss

    print('== TRAINED VARIABLES SUMMARIES ==')
    print(' - Generator variables:')
    for v in generator_variables:
        print(v.name)
    print(' - Discriminator variables:')
    for v in discriminator_variables:
        print(v.name)
    print(' - Classifier variables:')
    for v in classifier_variables:
        print(v.name)

    train_ops = {}
    train_ops['gen'] = gan_model.train_step(losses['gen']['joint'], generator_variables, optimizer_handle, learning_rate_gan)
    train_ops['disc'] = gan_model.train_step(losses['disc']['joint'], discriminator_variables, optimizer_handle, learning_rate_gan)
    train_ops['clf'] = gan_model.train_step(classifier_loss, classifier_variables, optimizer_handle, learning_rate_clf)

    return train_ops, losses
