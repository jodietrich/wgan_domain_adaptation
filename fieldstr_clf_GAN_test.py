__author__ = 'jdietric'

import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import shutil
import random
import importlib

import config.system as sys_config

import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import data_utils


#######################################################################

from experiments import residual_gen_bs2 as exp_config

#######################################################################

def generate_adapted_images(data, experiment, save_path):
    # extract images and indices of source/target images for the training and validation set
    images_train, source_images_train_ind, target_images_train_ind,\
    images_val, source_images_val_ind, target_images_val_ind = adni_data_loader.get_images_and_fieldstrength_indices(
        data, exp_config.source_field_strength, exp_config.target_field_strength)

    img_iter = ImageIterator((images_train, source_images_train_ind), (images_val, source_images_val_ind))

    # open save file from the selected experiment
    init_checkpoint_path = utils.get_latest_model_checkpoint_path(logdir, 'model.ckpt')
    logging.info('Checkpoint path: %s' % init_checkpoint_path)
    last_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1])  # plus 1 b/c otherwise starts with eval
    logging.info('Latest step was: %d' % last_step)

    generator = exp_config.generator

    z_sampler = data_utils.DataSampler(images_train, source_images_train_ind, images_val, source_images_val_ind)

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        im_s = exp_config.image_size

        training_placeholder = tf.placeholder(tf.bool, name='training_phase')

        # source image batch
        z_pl = tf.placeholder(tf.float32, [exp_config.batch_size, im_s[0], im_s[1], im_s[2], exp_config.n_channels], name='z')

        # generated fake image batch
        x_pl_ = generator(z_pl, training_placeholder)

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

        saver_latest.restore(sess, init_checkpoint_path)
        for source_img in img_iter:
            # classify source_img
            # generate image
            fake_img = sess.run(x_pl_, feed_dict={z_pl: z_img, training_placeholder: False})
            # classify fake_img


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
            input_folder=exp_config.data_root,
            preprocessing_folder=exp_config.preproc_folder,
            size=exp_config.image_size,
            target_resolution=exp_config.target_resolution,
            label_list = exp_config.label_list,
            force_overwrite=False
        )
    if not already_generated_images(experiment_name,image_saving_path):
        generate_adapted_images(data, experiment_name, image_saving_path)
    evaluate_images()


