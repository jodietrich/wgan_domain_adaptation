

__author__ = 'jdietric'

import logging

import itertools
import logging
import time
import numpy as np
import os.path
import tensorflow as tf
import shutil

import config.system as sys_config
import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import data_utils


#######################################################################
from experiments.gan import residual_gen_bs2 as gan_config
from experiments.fclf import jia_xi_net as fclf_config
#######################################################################

def generate_and_evaluate_adapted_images(data):
    # extract images and indices of source/target images for the training and validation set
    images_train, source_images_train_ind, target_images_train_ind,\
    images_val, source_images_val_ind, target_images_val_ind = adni_data_loader.get_images_and_fieldstrength_indices(
        data, gan_config.source_field_strength, gan_config.target_field_strength)

    # log file paths
    logdir_gan = os.path.join(sys_config.log_root, gan_config.experiment_name)
    logdir_fclf = os.path.join(sys_config.log_root, fclf_config.experiment_name)

    # open GAN save file from the selected experiment
    init_checkpoint_path_gan = utils.get_latest_model_checkpoint_path(logdir_gan, 'model.ckpt')
    logging.info('loading GAN')
    logging.info('Checkpoint path: %s' % init_checkpoint_path_gan)
    last_step_gan = int(init_checkpoint_path_gan.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step_gan)

    # open field strength classifier save file from the selected experiment
    init_checkpoint_path_fclf = utils.get_latest_model_checkpoint_path(logdir_fclf, 'model.ckpt')
    logging.info('loading field strength classifier')
    logging.info('Checkpoint path: %s' % init_checkpoint_path_fclf)
    last_step_fclf = int(init_checkpoint_path_fclf.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step_fclf)


    generator = gan_config.generator
    classifier = fclf_config.model_handle

    z_sampler = data_utils.DataSampler(images_train, source_images_train_ind, images_val, source_images_val_ind)

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        im_s = gan_config.image_size

        training_placeholder = tf.placeholder(tf.bool, name='training_phase')

        # source image batch
        z_pl = tf.placeholder(tf.float32, [gan_config.batch_size, im_s[0], im_s[1], im_s[2], gan_config.n_channels], name='z')

        # generated fake image batch
        x_pl_ = generator(z_pl, training_placeholder)

        # classification of the real source image and the fake target image
        logits_source_pl = classifier(z_pl, False, fclf_config.nlabels)
        logits_fake_pl = classifier(x_pl_, False, fclf_config.nlabels)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver_latest = tf.train.Saver(max_to_keep=3)
        saver_best_disc = tf.train.Saver(max_to_keep=3)  # disc loss is scaled negative EM distance

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        # Run the Op to initialize the variables.
        sess.run(init)

        saver_latest.restore(sess, init_checkpoint_path_gan)
        # create selectors
        train_source_sel, val_source_sel = utils.index_sets_to_selectors(source_images_train_ind, source_images_val_ind)

        # loops through all images from the source domain
        for source_img in itertools.chain(itertools.compress(images_train, train_source_sel),
                                          itertools.compress(images_val, val_source_sel)):
            # classify source_img
            source_logits = sess.run(logits_source_pl, feed_dict={z_pl: source_img, training_placeholder: False})
            # generate image
            fake_img = sess.run(x_pl_, feed_dict={z_pl: source_img, training_placeholder: False})
            # classify fake_img
            fake_logits = sess.run(logits_fake_pl, feed_dict={x_pl_: fake_img, training_placeholder: False})

            real_source_label = fclf_config.fs_label_list[fclf_config.field_strength_list.index(gan_config.source_field_strength)]
            predicted_source_label = fclf_config.fs_label_list[logits_source_pl.index(max(logits_source_pl))]
            predicted_fake_label = fclf_config.fs_label_list[logits_fake_pl.index(max(logits_fake_pl))]

            logging.info("NEW IMAGE")
            logging.info("real label of source image: " + str(real_source_label))
            logging.info("predicted label of source image: " + str(predicted_source_label))
            logging.info("predicted label of fake image: " + str(predicted_fake_label))





def import_images(path):
    pass

def evaluate_images(experiment):
    generated_images = import_images()

def already_generated_images(experiment, image_saving_path):
    return False


if __name__ == '__main__':
    # settings
    experiment_name = 'residual_identity_gen_bs2_std_disc_i2'
    fclf_experiment_name = 'jiaxi_net_only_diag_lr0.0001_flipaug_bn_mom0.99_fstr_all_data'
    image_saving_path = 'data/generated_images'

    # import data
    # TODO: import all data
    data = adni_data_loader.load_and_maybe_process_data(
            input_folder=gan_config.data_root,
            preprocessing_folder=gan_config.preproc_folder,
            size=gan_config.image_size,
            target_resolution=gan_config.target_resolution,
            label_list = gan_config.label_list,
            force_overwrite=False
        )

    generate_and_evaluate_adapted_images(data)


