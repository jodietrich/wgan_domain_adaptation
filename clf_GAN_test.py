# Authors:
# Jonathan Dietrich

# Test Classifiers using the test set

from test_utils import get_latest_checkpoint_and_log, evaluate_scores, build_clf_graph

__author__ = 'jdietric'

import itertools
import logging
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score
import csv
from collections import OrderedDict, Counter

import config.system as sys_config
import utils
import adni_data_loader
import adni_data_loader_all
import data_utils
import experiments.gan.standard_parameters as std_params
from batch_generator_list import iterate_minibatches
import test_utils


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


def build_gen_graph_old(img_tensor_shape, gan_config):
    generator = gan_config.generator
    graph_generator = tf.Graph()
    with graph_generator.as_default():
        # source image (batch size = 1)
        xs_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # generated fake image batch
        xf = generator(xs_pl, training=False)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()
        return graph_generator, xs_pl, xf, init, saver


def generate_and_evaluate_ad_classification(gan_experiment_path_list, clf_experiment_path, score_functions,
                                            image_saving_indices=set(), image_saving_path=None, max_batch_size=np.inf):
    """

    :param gan_experiment_path_list: list of GAN experiment paths to be evaluated. They must all have the same image settings and source/target field strengths as the classifier
    only gan experiments with the same source and target field strength are permitted
    :param clf_experiment_path: AD classifier used
    :param verbose: boolean. log all image classifications
    :param image_saving_indices: set of indices of the images to be saved
    :param image_saving_path: where to save the images. They are saved in subfolders for each experiment
    :return:
    """

    clf_config, logdir_clf = utils.load_log_exp_config(clf_experiment_path)

    # Load data
    data = adni_data_loader_all.load_and_maybe_process_data(
        input_folder=clf_config.data_root,
        preprocessing_folder=clf_config.preproc_folder,
        size=clf_config.image_size,
        target_resolution=clf_config.target_resolution,
        label_list=clf_config.label_list,
        offset=clf_config.offset,
        rescale_to_one=clf_config.rescale_to_one,
        force_overwrite=False
    )

    # extract images and indices of source/target images for the test set
    images_test = data['images_test']
    labels_test = data['diagnosis_test']
    ages_test = data['age_test']

    im_s = clf_config.image_size
    batch_size = min(clf_config.batch_size, std_params.batch_size, max_batch_size)
    logging.info('batch size %d is used for everything' % batch_size)
    img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]
    clf_remainder_batch_size = images_test.shape[0] % batch_size

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info("loading Alzheimer's disease classifier")
    graph_clf, image_pl, predictions_clf_op, init_clf_op, saver_clf = build_clf_graph(img_tensor_shape, clf_config)
    # logging.info("getting savepoint with the best cross entropy")
    # init_checkpoint_path_clf = get_latest_checkpoint_and_log(logdir_clf, 'model_best_xent.ckpt')
    logging.info("getting savepoint with the best f1 score")
    init_checkpoint_path_clf = get_latest_checkpoint_and_log(logdir_clf, 'model_best_diag_f1.ckpt')
    sess_clf = tf.Session(config=config, graph=graph_clf)
    sess_clf.run(init_clf_op)
    saver_clf.restore(sess_clf, init_checkpoint_path_clf)

    # make a separate graph for the last batch where the batchsize is smaller
    if clf_remainder_batch_size > 0:
        img_tensor_shape_gan_remainder = [clf_remainder_batch_size, im_s[0], im_s[1], im_s[2], 1]
        graph_clf_rem, image_pl_rem, predictions_clf_op_rem, init_clf_op_rem, saver_clf_rem = build_clf_graph(img_tensor_shape_gan_remainder, clf_config)
        sess_clf_rem = tf.Session(config=config, graph=graph_clf_rem)
        sess_clf_rem.run(init_clf_op_rem)
        saver_clf_rem.restore(sess_clf_rem, init_checkpoint_path_clf)

    # classifiy all real test images
    logging.info('classify all original images')
    real_pred = []
    for batch in iterate_minibatches(images_test,
                                     [labels_test, ages_test],
                                     batch_size=batch_size,
                                     exp_config=clf_config,
                                     map_labels_to_standard_range=False,
                                     shuffle_data=False,
                                     skip_remainder=False):
        # ignore the labels because data are in order, which means the label list in data can be used
        image_batch, [real_label, real_age] = batch

        current_batch_size = image_batch.shape[0]
        if current_batch_size < batch_size:
            clf_prediction_real = sess_clf_rem.run(predictions_clf_op_rem, feed_dict={image_pl_rem: image_batch})
        else:
            clf_prediction_real = sess_clf.run(predictions_clf_op, feed_dict={image_pl: image_batch})

        real_pred = real_pred + list(clf_prediction_real['label'])
        logging.info('new image batch')
        logging.info('ground truth labels: ' + str(real_label))
        logging.info('predicted labels: ' + str(clf_prediction_real['label']))

    gan_config0, logdir_gan0 = utils.load_log_exp_config(gan_experiment_path_list[0])

    source_indices = []
    target_indices = []
    source_true_labels = []
    target_true_labels = []
    for i, field_strength in enumerate(data['field_strength_test']):
        if field_strength == gan_config0.source_field_strength:
            source_indices.append(i)
            source_true_labels.append(labels_test[i])
        elif field_strength == gan_config0.target_field_strength:
            target_indices.append(i)
            target_true_labels.append(labels_test[i])

    # balance the test set
    (source_indices, source_true_labels), (
    target_indices, target_true_labels) = utils.balance_source_target(
        (source_indices, source_true_labels), (target_indices, target_true_labels), random_seed=0)
    source_pred = [pred for ind, pred in enumerate(real_pred) if ind in source_indices]
    target_pred = [pred for ind, pred in enumerate(real_pred) if ind in target_indices]

    assert len(source_pred) == len(source_true_labels)
    assert len(target_pred) == len(target_true_labels)

    # no unexpected labels
    assert all([label in clf_config.label_list for label in source_true_labels])
    assert all([label in clf_config.label_list for label in target_true_labels])
    assert all([label in clf_config.label_list for label in source_pred])
    assert all([label in clf_config.label_list for label in target_pred])

    num_source_images = len(source_indices)
    num_target_images = len(target_indices)

    source_label_count = Counter(source_true_labels)
    target_label_count = Counter(target_true_labels)

    logging.info('Data summary:')
    logging.info(' - Domains:')
    logging.info('number of source images: ' + str(num_source_images))
    logging.info('source label distribution ' + str(source_label_count))
    logging.info('number of target images: ' + str(num_target_images))
    logging.info('target label distribution ' + str(target_label_count))

    assert num_source_images == num_target_images
    assert source_label_count == target_label_count

    #2d image saving folder
    folder_2d = 'coronal_2d'
    image_saving_path2d = os.path.join(image_saving_path, folder_2d)
    utils.makefolder(image_saving_path2d)

    # save real images
    target_image_path = os.path.join(image_saving_path, 'target')
    source_image_path = os.path.join(image_saving_path, 'source')
    utils.makefolder(target_image_path)
    utils.makefolder(source_image_path)
    target_image_path2d = os.path.join(image_saving_path2d, 'target')
    source_image_path2d = os.path.join(image_saving_path2d, 'source')
    utils.makefolder(target_image_path2d)
    utils.makefolder(source_image_path2d)
    sorted_saving_indices = sorted(image_saving_indices)
    target_saving_indices = [target_indices[index] for index in sorted_saving_indices]
    for target_index in target_saving_indices:
        target_img_name = 'target_img_%.1fT_diag%d_ind%d' % (gan_config0.target_field_strength, labels_test[target_index], target_index)
        utils.save_image_and_cut(images_test[target_index], target_img_name, target_image_path, target_image_path2d)
        logging.info(target_img_name + ' saved')

    source_saving_indices = [source_indices[index] for index in sorted_saving_indices]
    for source_index in source_saving_indices:
        source_img_name = 'source_img_%.1fT_diag%d_ind%d' % (gan_config0.source_field_strength, labels_test[source_index], source_index)
        utils.save_image_and_cut(images_test[source_index], source_img_name, source_image_path,
                                 source_image_path2d)
        logging.info(source_img_name + ' saved')

    logging.info('source and target images saved')

    gan_remainder_batch_size = num_source_images % batch_size

    scores = {}
    for gan_experiment_path in gan_experiment_path_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_path)

        gan_experiment_name = gan_config.experiment_name

        # make sure the experiments all have the same configuration as the classifier
        assert gan_config.source_field_strength == gan_config0.source_field_strength
        assert gan_config.target_field_strength == gan_config0.target_field_strength
        assert gan_config.image_size == clf_config.image_size
        assert gan_config.target_resolution == clf_config.target_resolution
        assert gan_config.offset == clf_config.offset

        logging.info('\nGAN Experiment (%.1f T to %.1f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))
        logging.info(gan_config)
        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        # open the latest GAN savepoint
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator
        graph_generator, generator_img_pl, x_fake_op, init_gan_op, saver_gan = test_utils.build_gen_graph(img_tensor_shape, gan_config)

        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path = os.path.join(image_saving_path, gan_experiment_name)
        experiment_generate_path2d = os.path.join(image_saving_path2d, gan_experiment_name)
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path)
        utils.makefolder(experiment_generate_path2d)

        # make separate graphs for the last batch where the batchsize is smaller
        if clf_remainder_batch_size > 0:
            img_tensor_shape_gan_remainder = [gan_remainder_batch_size, im_s[0], im_s[1], im_s[2], 1]
            # classifier
            graph_clf_rem, image_pl_rem, predictions_clf_op_rem, init_clf_op_rem, saver_clf_rem = build_clf_graph(img_tensor_shape_gan_remainder, clf_config)
            sess_clf_rem = tf.Session(config=config, graph=graph_clf_rem)
            sess_clf_rem.run(init_clf_op_rem)
            saver_clf_rem.restore(sess_clf_rem, init_checkpoint_path_clf)

            # generator
            graph_generator_rem, generator_img_rem_pl, x_fake_op_rem, init_gan_op_rem, saver_gan_rem = \
                test_utils.build_gen_graph(img_tensor_shape_gan_remainder, gan_config)
            # Create a session for running Ops on the Graph.
            sess_gan_rem = tf.Session(config=config, graph=graph_generator_rem)
            # Run the Op to initialize the variables.
            sess_gan_rem.run(init_gan_op_rem)
            saver_gan_rem.restore(sess_gan_rem, init_checkpoint_path_gan)

        logging.info('image generation begins')
        generated_pred = []
        batch_beginning_index = 0
        # loops through all images from the source domain
        for batch in iterate_minibatches(images_test,
                                     [labels_test, ages_test],
                                     batch_size=batch_size,
                                     exp_config=clf_config,
                                     map_labels_to_standard_range=False,
                                     selection_indices=source_indices,
                                     shuffle_data=False,
                                     skip_remainder=False):
            # ignore the labels because data are in order, which means the label list in data can be used
            image_batch, [real_label, real_age] = batch

            current_batch_size = image_batch.shape[0]
            if current_batch_size < batch_size:
                fake_img = sess_gan_rem.run(x_fake_op_rem, feed_dict={generator_img_rem_pl: image_batch})
                # classify fake image
                clf_prediction_fake = sess_clf_rem.run(predictions_clf_op_rem, feed_dict={image_pl_rem: fake_img})
            else:
                fake_img = sess_gan.run(x_fake_op, feed_dict={generator_img_pl: image_batch})
                # classify fake image
                clf_prediction_fake = sess_clf.run(predictions_clf_op, feed_dict={image_pl: fake_img})

            generated_pred = generated_pred + list(clf_prediction_fake['label'])

            # save images
            current_source_indices = range(batch_beginning_index, batch_beginning_index + current_batch_size)

            # test whether minibatches are really iterated in order by checking if the labels are as expected
            assert [source_true_labels[i] for i in current_source_indices] == list(real_label)

            source_indices_to_save = image_saving_indices.intersection(set(current_source_indices))
            for source_index in source_indices_to_save:
                batch_index = source_index - batch_beginning_index
                # index of the image in the complete test data
                global_index = source_indices[source_index]
                generated_img_name = 'generated_img_%.1fT_diag%d_ind%d' % (gan_config.target_field_strength, labels_test[global_index], global_index)
                utils.save_image_and_cut(np.squeeze(fake_img[batch_index]), generated_img_name, experiment_generate_path, experiment_generate_path2d)
                logging.info(generated_img_name + ' saved')
                # save the difference g(xs)-xs
                corresponding_source_img = images_test[global_index]
                difference_image_gs = np.squeeze(fake_img[batch_index]) - corresponding_source_img
                difference_img_name = 'difference_img_%.1fT_diag%d_ind%d' % (gan_config.target_field_strength, labels_test[global_index], global_index)
                utils.save_image_and_cut(difference_image_gs, difference_img_name,
                                         experiment_generate_path, experiment_generate_path2d)
                logging.info(difference_img_name + ' saved')

            logging.info('new image batch')
            logging.info('ground truth labels: ' + str(real_label))
            logging.info('predicted labels for generated images: ' + str(clf_prediction_fake['label']))
            # no unexpected labels
            assert all([label in clf_config.label_list for label in clf_prediction_fake['label']])

            batch_beginning_index += current_batch_size
        logging.info('generated prediction for %s: %s' % (gan_experiment_name, str(generated_pred)))
        scores[gan_experiment_name] = evaluate_scores(source_true_labels, generated_pred, score_functions)

    logging.info('source prediction: ' + str(source_pred))
    logging.info('source ground truth: ' + str(source_true_labels))
    logging.info('target prediction: ' + str(target_pred))
    logging.info('target ground truth: ' + str(target_true_labels))

    scores['source_%.1fT' % gan_config0.source_field_strength] = evaluate_scores(source_true_labels, source_pred, score_functions)
    scores['target_%.1fT' % gan_config0.target_field_strength] = evaluate_scores(target_true_labels, target_pred, score_functions)

    return scores



def generate_and_evaluate_fieldstrength_classification(gan_experiment_path_list, fclf_experiment_path, verbose=True,
                                                       num_saved_images=0, image_saving_path=None):
    """Old function without the balanced test set

    :param gan_experiment_path_list:
    :param fclf_experiment_path:
    :param verbose:
    :param num_saved_images:
    :param image_saving_path:
    :return:
    """
    # bigger does not work currently (because of the statistics)
    batch_size = 1

    fclf_config, logdir_fclf = utils.load_log_exp_config(fclf_experiment_path)

    im_s = fclf_config.image_size
    img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info('loading field strength classifier')
    graph_fclf, fclf_pl, predictions_fclf_op, init_fclf_op, saver_fclf = build_clf_graph(img_tensor_shape, fclf_config)
    init_checkpoint_path_fclf = get_latest_checkpoint_and_log(logdir_fclf, 'model_best_diag_f1.ckpt')
    sess_fclf = tf.Session(config=config, graph=graph_fclf)
    sess_fclf.run(init_fclf_op)
    saver_fclf.restore(sess_fclf, init_checkpoint_path_fclf)

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
    for gan_experiment_path in gan_experiment_path_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_path)
        gan_experiment_name = gan_config.experiment_name
        logging.info('\nGAN Experiment (%f T to %f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))

        # extract images and indices of source/target images for the training and validation set
        images_train, source_images_train_ind, target_images_train_ind,\
        images_val, source_images_val_ind, target_images_val_ind = data_utils.get_images_and_fieldstrength_indices(
            data, gan_config.source_field_strength, gan_config.target_field_strength)

        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator and the classifier respectively
        graph_generator, gan_pl, x_fake_op, init_gan_op, saver_gan = test_utils.build_gen_graph(img_tensor_shape, gan_config)


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

        source_label, target_label = utils.fstr_to_label([gan_config.source_field_strength, gan_config.target_field_strength],
                                                             fclf_config.field_strength_list, fclf_config.fs_label_list)

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
    gan_experiment_list_s3 = [
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s3_final_i1',
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_no_noise_all_small_data_1e4l1_s3_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s3_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_no_noise_all_small_data_1e4l1_s3_final_i1',
    ]

    gan_experiment_list_s15 = [
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s15_final_i1',
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_no_noise_all_small_data_1e4l1_s15_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s15_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_no_noise_all_small_data_1e4l1_s15_final_i1',
    ]

    joint_experiment_list_s3 = [
        'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1_cont',
        'joint_genval_gan_residual_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1_cont',
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1',
        'joint_genval_gan_residual_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1',
    ]

    joint_experiment_list_s15 = [
        'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s15_bs6_i1_cont',
        'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s15_bs6_i1_cont',
        'joint_genval_gan_residual_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s15_bs6_i1_cont',
    ]

    joint_beta_test_list = [
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e3_all_small_final_s3_bs6_i1',
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1',
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e7_all_small_final_s3_bs6_i1'
    ]

    gan_experiment_list = gan_experiment_list_s15  # <---------------------------------
    results_save_file_name = 'gan_experiments_s15_clf_test_f1_sel.csv'  # <---------------------------------
    results_save_folder = 'results/final/gan_test_target_clf'

    results_save_path = os.path.join(sys_config.project_root, results_save_folder, results_save_file_name)

    # clf_experiment_name = 'adni_clf_bs20_domains_t15_data_final_i1'  # <---------------------------------
    clf_experiment_name = 'adni_clf_bs20_domains_s3_data_final_i1'  # <---------------------------------
    clf_log_root = os.path.join(sys_config.log_root, 'adni_clf/final')
    gan_log_root = os.path.join(sys_config.log_root, 'gan/final')  # <---------------------------------
    image_saving_path = os.path.join(sys_config.project_root,'data/generated_images/final/all_experiments')
    image_saving_indices = set(range(0, 220, 5))

    # put paths for experiments together
    clf_log_path = os.path.join(clf_log_root, clf_experiment_name)
    gan_log_path_list = [os.path.join(gan_log_root, gan_name) for gan_name in gan_experiment_list]

    # import config file for field strength classifier
    logging.info('Classifier used: ' + clf_experiment_name)

    # what is scored for source, target and generated images
    score_functions = {'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=2, average='binary'),
                       'recall':  lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=2, average='binary'),
                       'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, pos_label=2, average='binary')
    }

    clf_scores = generate_and_evaluate_ad_classification(gan_experiment_path_list=gan_log_path_list,
                                                         clf_experiment_path=clf_log_path,
                                                         score_functions=score_functions,
                                                         image_saving_indices=image_saving_indices,
                                                         image_saving_path=image_saving_path, max_batch_size=np.inf)

    # function to get the f1 score from an element of clf_scores.items()
    get_f1_score = lambda dict_key: clf_scores[dict_key]['f1']

    # logging.info('key function test')
    # logging.info(clf_scores.items()[0])
    # logging.info(get_f1_score(clf_scores.items()[0]))

    scores_string = utils.string_dict_in_order(clf_scores, key_function=get_f1_score)
    logging.info('FINAL SUMMARY:\nordered by f1 score in descending order\n' + scores_string)

    # gives the name of the experiment with the best f1 score on the generated images
    best_experiment = max(clf_scores, key=get_f1_score)
    best_score = get_f1_score(best_experiment)
    logging.info('The best experiment was %s with f1 score %f for the generated images' % (best_experiment, best_score))

    # get scores ordered in the right way
    source_key = set(key for key in clf_scores if key.startswith('source'))
    target_key = set(key for key in clf_scores if key.startswith('target'))

    assert len(source_key) == 1
    assert len(target_key) == 1

    exp_keys = sorted([key for key in clf_scores if key not in source_key.union(target_key)])
    exp_keys = list(source_key) + exp_keys + list(target_key)
    logging.info(exp_keys)

    # save the result as a csv file
    with open(results_save_path, 'w+', newline='') as csvfile:
        fieldnames = ['experiment name', 'f1', 'recall', 'precision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for curr_exp_name in exp_keys:
            logging.info(curr_exp_name)
            logging.info(clf_scores[curr_exp_name])
            row_dict = {'experiment name': curr_exp_name}
            row_dict.update(clf_scores[curr_exp_name])
            writer.writerow(row_dict)

        # write classifier experiment name at the last row
        writer.writerow({'experiment name': clf_experiment_name})




