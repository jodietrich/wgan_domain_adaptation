# test a given classifier on either 1.5T, 3T or both
# print f1 score, accuracy, precision and the confusion matrix
# make it possible to test either the best validation f1 score or loss checkpoints
import test_utils

__author__ = 'jdietric'

import itertools
import logging
import time
import numpy as np
import os
import tensorflow as tf
import shutil
from importlib.machinery import SourceFileLoader
import sklearn.metrics
import operator

import config.system as sys_config
import gan_model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import adni_data_loader_all
import data_utils
from clf_model_multitask import predict
import experiments.gan.standard_parameters as std_params
from batch_generator_list import iterate_minibatches
import clf_GAN_test
from collections import OrderedDict


def classifier_test(clf_experiment_path, score_functions, batch_size=1):
    """

    :param clf_experiment_path: AD classifier used
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

    logging.info('batch size %d is used for classifier' % batch_size)
    img_tensor_shape = [None] + list(clf_config.image_size) + [1]

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info("loading Alzheimer's disease classifier")
    graph_clf, image_pl, predictions_clf_op, init_clf_op, saver_clf = test_utils.build_clf_graph(img_tensor_shape, clf_config)
    logging.info("getting savepoint with the best f1 score")
    checkpoint_file_name = 'model_best_diag_f1.ckpt'
    # logging.info("getting savepoint with the best cross entropy")
    # checkpoint_file_name = 'model_best_xent'
    init_checkpoint_path_clf, latest_step = utils.get_latest_checkpoint_and_step(logdir_clf, checkpoint_file_name)
    # logging.info("getting savepoint with the best f1 score")
    # init_checkpoint_path_clf = get_latest_checkpoint_and_log(logdir_clf, 'model_best_diag_f1.ckpt')
    sess_clf = tf.Session(config=config, graph=graph_clf)
    sess_clf.run(init_clf_op)  # probably not necessary
    saver_clf.restore(sess_clf, init_checkpoint_path_clf)

    # classifiy all real test images
    logging.info('classify all test images')
    all_predictions = []
    ground_truth_labels = []
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
        clf_prediction_batch = sess_clf.run(predictions_clf_op, feed_dict={image_pl: image_batch})

        all_predictions = all_predictions + list(clf_prediction_batch['label'])
        ground_truth_labels = ground_truth_labels + list(real_label)
        logging.info('new image batch')
        logging.info('ground truth labels: ' + str(real_label))
        logging.info('predicted labels: ' + str(clf_prediction_batch['label']))

    # check that the data has really been iterated in order and in full
    assert np.array_equal(ground_truth_labels, labels_test)

    source_indices = []
    target_indices = []
    source_true_labels = []
    source_pred = []
    target_true_labels = []
    target_pred = []
    for i, field_strength in enumerate(data['field_strength_test']):
        if field_strength == clf_config.source_field_strength:
            source_indices.append(i)
            source_true_labels.append(labels_test[i])
            source_pred.append(all_predictions[i])
        elif field_strength == clf_config.target_field_strength:
            target_indices.append(i)
            target_true_labels.append(labels_test[i])
            target_pred.append(all_predictions[i])

    # check that the source and target images together are all images
    all_indices = source_indices + target_indices
    all_indices.sort()
    assert np.array_equal(all_indices, range(images_test.shape[0]))

    # no unexpected labels
    assert all([label in clf_config.label_list for label in source_true_labels])
    assert all([label in clf_config.label_list for label in target_true_labels])
    assert all([label in clf_config.label_list for label in source_pred])
    assert all([label in clf_config.label_list for label in target_pred])

    num_source_images = len(source_indices)
    num_target_images = len(target_indices)

    # count how many there are of each label
    label_count = {label: 0 for label in clf_config.label_list}
    source_label_count = label_count.copy()
    target_label_count = label_count.copy()
    for label in labels_test:
        label_count[label] += 1
    for label in source_true_labels:
        source_label_count[label] += 1
    for label in target_true_labels:
        target_label_count[label] += 1

    logging.info('Data summary:')
    logging.info(' - Images:')
    logging.info(images_test.shape)
    logging.info(images_test.dtype)
    logging.info(' - Labels:')
    logging.info(labels_test.shape)
    logging.info(labels_test.dtype)
    logging.info('number of images for each label')
    logging.info(label_count)
    logging.info(' - Domains:')
    logging.info('number of source images: ' + str(num_source_images))
    logging.info('source label distribution ' + str(source_label_count))
    logging.info('number of target images: ' + str(num_target_images))
    logging.info('target label distribution ' + str(target_label_count))

    scores = {}

    logging.info('source prediction: ' + str(source_pred))
    logging.info('source ground truth: ' + str(source_true_labels))
    logging.info('target prediction: ' + str(target_pred))
    logging.info('target ground truth: ' + str(target_true_labels))

    scores[clf_config.source_field_strength] = test_utils.evaluate_scores(source_true_labels, source_pred, score_functions)
    scores[clf_config.target_field_strength] = test_utils.evaluate_scores(target_true_labels, target_pred, score_functions)
    scores['all data'] = test_utils.evaluate_scores(labels_test, all_predictions, score_functions)

    # dictionary sorted by key
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: str(t[0])))

    return sorted_scores, latest_step



def test_multiple_classifiers(classifier_exp_list, joint):
    for clf_experiment_name in classifier_exp_list:
        if joint:
            clf_log_root = os.path.join(sys_config.log_root, 'joint/final')
        else:
            clf_log_root = os.path.join(sys_config.log_root, 'adni_clf/final')

        # put paths for experiments together
        clf_log_path = os.path.join(clf_log_root, clf_experiment_name)

        # import config file for field strength classifier
        logging.info('Classifier used: ' + clf_experiment_name)

        # what is scored for source, target and generated images
        score_functions = OrderedDict([
            ('f1', lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, pos_label=2, average='binary')),
            ('recall',
             lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, pos_label=2, average='binary')),
            ('precision',
             lambda y_true, y_pred: sklearn.metrics.precision_score(y_true, y_pred, pos_label=2, average='binary')),
            ('confusion matrix',
             lambda y_true, y_pred: sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 2]))])

        # confusion matrix = [[tn, fp], [fn, tp]]

        clf_scores, latest_step = classifier_test(clf_experiment_path=clf_log_path,
                                     score_functions=score_functions,
                                     batch_size=20)

        logging.info('results for ' + str(clf_experiment_name))
        logging.info(clf_scores)

        clf_score_string = nested_dict_multi_line_string(clf_scores)

        # write results to a file
        experiment_file_name = clf_experiment_name + '_step%d' % latest_step
        result_file_path = os.path.join(sys_config.project_root, 'results/final/clf_test', experiment_file_name)
        # overwrites the old file if there is already a file with this name
        with open(result_file_path, "w") as result_file:
            result_file.write(clf_experiment_name + '\n')
            result_file.write('step: %d\n' % latest_step)
            result_file.write(clf_score_string)


def nested_dict_multi_line_string(dict):
    # makes a string from a dict of dicts with each value of the inner dicts in a separate line
    # and each inner dict separated by an empty line
    outer_lines = []
    for outer_key, inner_dict in dict.items():
        outer_string = str(outer_key) + ':\n'
        for inner_key, value in inner_dict.items():
            inner_string = str(inner_key) + ': %s\n' % str(value)
            outer_string += inner_string
        outer_string += '\n'
        outer_lines.append(outer_string)

    return '\n'.join(outer_lines)


if __name__ == '__main__':
    classifier_experiment_list1 = [
        'adni_clf_bs20_domains_s_bousmalis_gen_1e4l1_no_noise_s3_data_final_i1',
        'adni_clf_cropdata_allconv_yesrescale_bs20_bn_all_both_domains_s3_data_final_i1',
        'adni_clf_cropdata_allconv_yesrescale_bs20_bn_all_source3_data_final_i1',
        'adni_clf_cropdata_allconv_yesrescale_bs20_bn_all_target15_data_final_i1'
    ]
    joint_list1 = [
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e3_all_small_final_s3_bs6_i1',
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1',
        'joint_fixed_clf_allconv_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e7_all_small_final_s3_bs6_i1'
    ]
    classifier_experiment_list2 = [
        'adni_clf_bs20_domains_t15_data_final_i1'
    ]
    classifier_experiment_list3 = [
        'adni_clf_bs20_domains_all_data_final_i1',
        'adni_clf_bs20_domains_s3_data_final_i1',
        'adni_clf_bs20_domains_s3_gen_bousmalis_1e4l1_10_noise_s3_data_final_i1'
    ]
    all_clf_list = classifier_experiment_list1 + classifier_experiment_list2 + classifier_experiment_list3

    test_multiple_classifiers(all_clf_list, joint=False)






