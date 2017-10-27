

__author__ = 'jdietric'

import itertools
import logging
import time
import numpy as np
import os
import tensorflow as tf
import shutil
from importlib.machinery import SourceFileLoader

import config.system as sys_config
import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import data_utils
from model_multitask import predict


def generate_and_evaluate_adapted_images(data, gan_config, fclf_config):
    """
    :param data: hdf5 file handle with ADNI data
    :param gan_config: SourceFileLoader from importlib.machinery for gan config file
    :param fclf_config: SourceFileLoader from importlib.machinery for fclf config file
    :return: nothing
    """


    # extract images and indices of source/target images for the training and validation set
    images_train, source_images_train_ind, target_images_train_ind,\
    images_val, source_images_val_ind, target_images_val_ind = adni_data_loader.get_images_and_fieldstrength_indices(
        data, gan_config.source_field_strength, gan_config.target_field_strength)

    # open GAN save file from the selected experiment
    init_checkpoint_path_gan = utils.get_latest_model_checkpoint_path(logdir_gan, 'model.ckpt')
    logging.info('loading GAN')
    logging.info('Checkpoint path: %s' % init_checkpoint_path_gan)
    last_step_gan = int(init_checkpoint_path_gan.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step_gan)

    # open field strength classifier save file from the selected experiment
    init_checkpoint_path_fclf = utils.get_latest_model_checkpoint_path(logdir_fclf, 'model_best_xent.ckpt')
    logging.info('loading field strength classifier')
    logging.info('Checkpoint path: %s' % init_checkpoint_path_fclf)
    last_step_fclf = int(init_checkpoint_path_fclf.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step_fclf)


    generator = gan_config.generator
    # classifier = lambda images: fclf_config.model_handle(images=images, training=False, nlabels=fclf_config.nlabels,
    #                                                      bn_momentum=fclf_config.bn_momentum, scope_reuse=True)

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
        # TODO: figure out reuse issue
        with tf.variable_scope('prediction') as scope:
            scope.reuse_variables()
            source_predicted_label, source_softmax, _ = predict(z_pl, fclf_config)
            fake_predicted_label, fake_softmax, _ = predict(x_pl_, fclf_config)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver_latest_gan = tf.train.Saver()
        saver_best_fclf = tf.train.Saver()  # disc loss is scaled negative EM distance

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        # Run the Op to initialize the variables.
        sess.run(init)

        saver_latest_gan.restore(sess, init_checkpoint_path_gan)
        saver_best_fclf.restore(sess, init_checkpoint_path_fclf)
        # create selectors
        train_source_sel, val_source_sel = utils.index_sets_to_selectors(source_images_train_ind, source_images_val_ind)

        # s for source, t for target. First the prediction on the source image, then the prediction on the generated image
        prediction_count = {'ss': 0, 'st': 0, 'ts': 0, 'tt': 0}
        # loops through all images from the source domain
        for source_img in itertools.chain(itertools.compress(images_train, train_source_sel),
                                          itertools.compress(images_val, val_source_sel)):
            # classify source_img
            source_prediction, source_sm_prob = sess.run([source_predicted_label, source_softmax], feed_dict={z_pl: source_img, training_placeholder: False})
            # generate image
            fake_img = sess.run(x_pl_, feed_dict={z_pl: source_img, training_placeholder: False})
            # classify fake_img
            fake_prediction, fake_sm_prob = sess.run([fake_predicted_label, fake_softmax], feed_dict={x_pl_: fake_img, training_placeholder: False})

            source_label = fclf_config.fs_label_list[fclf_config.field_strength_list.index(gan_config.source_field_strength)]

            # record occurences of the four possible combinations of source_prediction and fake_prediction
            if source_prediction == source_label:
                if fake_prediction == source_label:
                    prediction_count['ss'] += 1
                else:
                    prediction_count['st'] += 1
            else:
                if fake_prediction == source_label:
                    prediction_count['ts'] += 1
                else:
                    prediction_count['tt'] += 1


            logging.info("NEW IMAGE")
            logging.info("real label of source image: " + str(source_label))
            logging.info("predicted label of source image: " + str(source_prediction))
            logging.info("predicted label of fake image: " + str(fake_prediction))

        log_stats(prediction_count)


def log_stats(prediction_count):
    total_count = 0
    for key in prediction_count:
        total_count += prediction_count[key]
    logging.info('SUMMARY')
    logging.info('fraction of generated pictures classified as target domain images: ' + str((prediction_count['st'] + prediction_count['tt']))/total_count)
    logging.info('total number of pictures processed: ' + str(total_count))

    logging.info('statistics with pictures where the source image was correctly classified as a source domain image:')
    source_real_count = prediction_count['ss'] + prediction_count['st']
    logging.info('number of images: ' + str(source_real_count))
    if source_real_count > 0:
        logging.info('fraction of generated pictures classified as target domain images: ' + str(prediction_count['st']/source_real_count))

    logging.info('statistics with pictures where the source image was incorrectly classified as a target domain image:')
    target_real_count = prediction_count['ts'] + prediction_count['tt']
    logging.info('number of images: ' + str(target_real_count))
    if target_real_count > 0:
        logging.info('fraction of generated pictures classified as target domain images: ' + str(prediction_count['tt']/target_real_count))


if __name__ == '__main__':
    # settings
    gan_experiment_name = 'residual_identity_gen_bs2_std_disc_i2'
    fclf_experiment_name = 'fclf_jiaxi_net_small_data'
    image_saving_path = 'data/generated_images'

    # import config files
    gan_config = utils.load_log_exp_config(gan_experiment_name)
    fclf_config = utils.load_log_exp_config(fclf_experiment_name)

    # import data
    data = adni_data_loader.load_and_maybe_process_data(
            input_folder=gan_config.data_root,
            preprocessing_folder=gan_config.preproc_folder,
            size=gan_config.image_size,
            target_resolution=gan_config.target_resolution,
            label_list = (0, 1, 2),
            force_overwrite=False
        )

    generate_and_evaluate_adapted_images(data, gan_config, fclf_config)


