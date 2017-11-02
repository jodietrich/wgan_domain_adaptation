

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
import adni_data_loader_all
import data_utils
from model_multitask import predict
import experiments.gan.standard_parameters as std_params


def get_latest_checkpoint_and_log(logdir, filename):
    init_checkpoint_path = utils.get_latest_model_checkpoint_path(logdir, filename)
    logging.info('Checkpoint path: %s' % init_checkpoint_path)
    last_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step)
    return init_checkpoint_path

def log_stats_fclf(prediction_count, source_label, target_label):
    total_count = 0
    for key in prediction_count:
        total_count += prediction_count[key]
    ss = prediction_count[(source_label, source_label)]
    st = prediction_count[(source_label, target_label)]
    ts = prediction_count[(target_label, source_label)]
    tt = prediction_count[(target_label, target_label)]
    score = (st + tt)/total_count
    logging.info('SUMMARY')
    logging.info('fraction of generated pictures classified as target domain images: ' + str(score))
    logging.info('total number of pictures processed: ' + str(total_count))

    logging.info('statistics with pictures where the source image was correctly classified as a source domain image:')
    source_real_count = ss + tt
    logging.info('number of images: ' + str(source_real_count))
    if source_real_count > 0:
        logging.info('fraction of generated pictures classified as target domain images: ' + str(st/source_real_count))

    logging.info('statistics with pictures where the source image was incorrectly classified as a target domain image:')
    target_real_count = ts + tt
    logging.info('number of images: ' + str(target_real_count))
    if target_real_count > 0:
        logging.info('fraction of generated pictures classified as target domain images: ' + str(tt/target_real_count))
    return score


def build_clf_graph(img_tensor_shape, clf_config):
    graph_classifier = tf.Graph()
    with graph_classifier.as_default():
        # Generate placeholders for the images and labels.
        training_clf_pl = tf.placeholder(tf.bool, name='training_phase')

        # source image (batch size = 1)
        xs_clf_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # generated fake image batch
        xf_clf_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # classification of the real source image and the fake target image
        source_predicted_label, source_softmax, _ = predict(xs_clf_pl, clf_config)
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        fake_predicted_label, fake_softmax, _ = predict(xf_clf_pl, clf_config)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()  # disc loss is scaled negative EM distance
        placeholders = {'source_img': xs_clf_pl, 'fake_img': xf_clf_pl, 'training': training_clf_pl}
        predictions = {'source_label': source_predicted_label[0], 'source_softmax': source_softmax,
                       'fake_label': fake_predicted_label[0], 'fake_softmax': fake_softmax}
        return graph_classifier, placeholders, predictions, init, saver


def build_gen_graph(img_tensor_shape, gan_config):
    generator = gan_config.generator
    graph_generator = tf.Graph()
    with graph_generator.as_default():
        training_pl = tf.placeholder(tf.bool, name='training_phase')

        # source image (batch size = 1)
        xs_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # generated fake image batch
        xf = generator(xs_pl, training_pl)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()
        placeholders = {'source_img': xs_pl, 'training': training_pl}
        return graph_generator, placeholders, xf, init, saver


def generate_and_evaluate_ad_classification(gan_experiment_list, clf_experiment_name, verbose=True, num_saved_images=0, image_saving_path=None):
    """

    :param gan_experiment_list: list of GAN experiment names to be evaluated. They must all have the same image settings
    :param clf_experiment_name: AD classifier used
    :param verbose: boolean. log all image classifications
    :param num_saved_images: how many images to save as nii files. 0 if none should be saved
    :param image_saving_path: where to save the images. They are saved in subfolders for each experiment
    :return:
    """
    # bigger does not work currently (because of the statistics)
    batch_size = 1

    clf_config, logdir_clf = utils.load_log_exp_config(clf_experiment_name)

    im_s = clf_config.image_size
    img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info("loading Alzheimer's disease classifier")
    graph_clf, clf_pl, predictions_clf_op, init_clf_op, saver_clf = build_clf_graph(img_tensor_shape, clf_config)
    init_checkpoint_path_clf = get_latest_checkpoint_and_log(logdir_clf, 'model_best_xent.ckpt')
    sess_clf = tf.Session(config=config, graph=graph_clf)
    sess_clf.run(init_clf_op)
    saver_clf.restore(sess_clf, init_checkpoint_path_clf)

    #TODO: report the f1 score of the classifier on the validation set from the training run
    # graph_fclf.get_tensor_by_name('?')

    # import data
    data = adni_data_loader_all.load_and_maybe_process_data(
            input_folder=clf_config.data_root,
            preprocessing_folder=clf_config.preproc_folder,
            size=clf_config.image_size,
            target_resolution=clf_config.target_resolution,
            label_list = (0, 1, 2),
            force_overwrite=False
        )

     # extract images and indices of source/target images for the test set
    images_test = data['images_test']
    labels_test = data['diagnosis_test']
    ages_test = data['age_test']

    num_images = images_test.shape[0]
    logging.info('there are %d test images')

    scores = {}
    for gan_experiment_name in gan_experiment_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_name)
        logging.info('\nGAN Experiment (%f T to %f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))

        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator and the classifier respectively
        graph_generator, gan_pl, x_fake_op, init_gan_op, saver_gan = build_gen_graph(img_tensor_shape, gan_config)

        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path = os.path.join(image_saving_path, gan_experiment_name)
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path)

        # create selectors
        source_selector = [field_strength == gan_config.source_field_strength for field_strength in data['field_strength_test']]

        num_source_images = np.sum([1 for is_source in source_selector if is_source])

        # create a dictionary with labellist^3 as keys and all values initialized as 0
        # to count all possible combinations of (ground truth label, predicted label of source image, predicted label of generated image)
        prediction_count = {combination: 0 for combination in itertools.product(clf_config.label_list, repeat=3)}
        # loops through all images from the source domain
        for img_num, source_img, label in enumerate(itertools.compress(itertools.izip(images_test, labels_test), source_selector)):
            source_image_input = np.reshape(source_img, img_tensor_shape)
            # generate image
            feeddict_gan = {gan_pl['source_img']: source_image_input, gan_pl['training']: False}
            fake_img = sess_gan.run(x_fake_op, feed_dict=feeddict_gan)
            # classify images
            feeddict_fclf = {clf_pl['source_img']: source_image_input, clf_pl['fake_img']: fake_img, clf_pl['training']: False}
            fclf_predictions_dict = sess_clf.run(predictions_clf_op, feed_dict=feeddict_fclf)

            # save images
            if img_num < num_saved_images:
                source_img_name = 'source_img_%.1fT_%d.nii.gz' % (gan_config.source_field_strength, img_num)
                generated_img_name = 'generated_img_%.1fT_%d.nii.gz' % (gan_config.target_field_strength, img_num)
                utils.create_and_save_nii(np.squeeze(source_img), os.path.join(experiment_generate_path, source_img_name))
                utils.create_and_save_nii(np.squeeze(fake_img), os.path.join(experiment_generate_path, generated_img_name))
                logging.info('images saved')


            # record occurences of the four possible combinations of source_prediction and fake_prediction
            label_tuple = (label, fclf_predictions_dict['source_label'], fclf_predictions_dict['fake_label'])
            prediction_count[label_tuple] += 1

            if verbose:
                logging.info("NEW IMAGE")
                logging.info("ground truth label of source image: " + str(label))
                logging.info("predictions: " + str(fclf_predictions_dict))

        true_predictions = np.sum([prediction_count[labels] for labels in prediction_count if labels[0] == labels[2]])
        scores[gan_experiment_name] = true_predictions/num_source_images

    return scores


def generate_and_evaluate_fieldstrength_classification(gan_experiment_list, fclf_experiment_name, verbose=True, num_saved_images=0, image_saving_path=None):
    """

    :param gan_experiment_list:
    :param fclf_experiment_name:
    :param verbose:
    :param num_saved_images:
    :param image_saving_path:
    :return:
    """
    # bigger does not work currently (because of the statistics)
    batch_size = 1

    fclf_config, logdir_fclf = utils.load_log_exp_config(fclf_experiment_name)

    im_s = fclf_config.image_size
    img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info('loading field strength classifier')
    graph_fclf, fclf_pl, predictions_fclf_op, init_fclf_op, saver_fclf = build_clf_graph(img_tensor_shape, fclf_config)
    init_checkpoint_path_fclf = get_latest_checkpoint_and_log(logdir_fclf, 'model_best_xent.ckpt')
    sess_fclf = tf.Session(config=config, graph=graph_fclf)
    sess_fclf.run(init_fclf_op)
    saver_fclf.restore(sess_fclf, init_checkpoint_path_fclf)

    #TODO: report the f1 score of the fieldstrenght_classifier on the validation set from the training run
    # graph_fclf.get_tensor_by_name('?')

    # import data
    data = adni_data_loader.load_and_maybe_process_data(
            input_folder=fclf_config.data_root,
            preprocessing_folder=fclf_config.preproc_folder,
            size=fclf_config.image_size,
            target_resolution=fclf_config.target_resolution,
            label_list = (0, 1, 2),
            force_overwrite=False
        )

    scores = {}
    for gan_experiment_name in gan_experiment_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_name)
        logging.info('\nGAN Experiment (%f T to %f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))

        # extract images and indices of source/target images for the training and validation set
        images_train, source_images_train_ind, target_images_train_ind,\
        images_val, source_images_val_ind, target_images_val_ind = adni_data_loader.get_images_and_fieldstrength_indices(
            data, gan_config.source_field_strength, gan_config.target_field_strength)

        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator and the classifier respectively
        graph_generator, gan_pl, x_fake_op, init_gan_op, saver_gan = build_gen_graph(img_tensor_shape, gan_config)


        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path = os.path.join(image_saving_path, gan_experiment_name)
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path)

        # create selectors
        train_source_sel, val_source_sel = utils.index_sets_to_selectors(source_images_train_ind, source_images_val_ind)

        # s for source, t for target. First the prediction on the source image, then the prediction on the generated image
        prediction_count = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        # loops through all images from the source domain
        for img_num, source_img in enumerate(itertools.chain(itertools.compress(images_train, train_source_sel),
                                          itertools.compress(images_val, val_source_sel))):
            source_image_input = np.reshape(source_img, img_tensor_shape)
            # generate image
            feeddict_gan = {gan_pl['source_img']: source_image_input, gan_pl['training']: False}
            fake_img = sess_gan.run(x_fake_op, feed_dict=feeddict_gan)
            # classify images
            feeddict_fclf = {fclf_pl['source_img']: source_image_input, fclf_pl['fake_img']: fake_img, fclf_pl['training']: False}
            fclf_predictions_dict = sess_fclf.run(predictions_fclf_op, feed_dict=feeddict_fclf)
            source_label, target_label = utils.fstr_to_label([gan_config.source_field_strength, gan_config.target_field_strength],
                                                             fclf_config.field_strength_list, fclf_config.fs_label_list)


            # save images
            if img_num < num_saved_images:
                source_img_name = 'source_img_%.1fT_%d.nii.gz' % (gan_config.source_field_strength, img_num)
                generated_img_name = 'generated_img_%.1fT_%d.nii.gz' % (gan_config.target_field_strength, img_num)
                utils.create_and_save_nii(np.squeeze(source_img), os.path.join(experiment_generate_path, source_img_name))
                utils.create_and_save_nii(np.squeeze(fake_img), os.path.join(experiment_generate_path, generated_img_name))
                logging.info('images saved')


            # record occurences of the four possible combinations of source_prediction and fake_prediction
            label_tuple = (fclf_predictions_dict['source_label'], fclf_predictions_dict['fake_label'])
            prediction_count[label_tuple] += 1

            if verbose:
                logging.info("NEW IMAGE")
                logging.info("real label of source image: " + str(source_label))
                logging.info("predictions: " + str(fclf_predictions_dict))

        scores[gan_experiment_name] = log_stats_fclf(prediction_count, source_label, target_label)

    return scores



if __name__ == '__main__':
    # settings
    gan_experiment_list = [
        'residual_identity_gen_bs1_std_disc_i1',
        'residual_identity_gen_bs2_std_disc_bn_i1',
        'residual_identity_gen_bs2_std_disc_i1',
        'residual_identity_gen_bs2_std_disc_i2',
        'std_cnn_identity_gen_v5'
    ]
    fclf_experiment_name = 'fclf_jiaxi_net_small_data'
    image_saving_path = os.path.join(sys_config.project_root,'data/generated_images')

    # import config file for field strength classifier
    logging.info('Classifier used: ' + fclf_experiment_name)

    fclf_scores = generate_and_evaluate_fieldstrength_classification(gan_experiment_list, fclf_experiment_name, verbose=True, num_saved_images=10, image_saving_path=image_saving_path)

    logging.info('FINAL SUMMARY:\nFraction of generated images classified as from the target domain (score):')
    logging.info(fclf_scores)
    # gives the name of the experiment with the largest score (dictionary iterates over keys)
    best_experiment = max(fclf_scores, key=fclf_scores.get)
    best_score = fclf_scores[best_experiment]
    logging.info('The best experiment was %s with score %f' % (best_experiment, best_score))





