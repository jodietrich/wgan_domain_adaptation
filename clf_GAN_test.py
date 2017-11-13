

__author__ = 'jdietric'

import itertools
import logging
import time
import numpy as np
import os
import tensorflow as tf
import shutil
from importlib.machinery import SourceFileLoader
from sklearn.metrics import f1_score, recall_score, precision_score
import operator

import config.system as sys_config
import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import adni_data_loader_all
import data_utils
from model_multitask import predict
import experiments.gan.standard_parameters as std_params
from batch_generator_list import iterate_minibatches


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

def evaluate_scores(true_labels, prediction, measures_dict):
    scores_one_exp = {}
    for measure_name, measure in measures_dict.items():
        logging.info('evaluating ' + measure_name)
        logging.info(measure)
        scores_one_exp[measure_name] = measure(y_true = np.asarray(true_labels), y_pred = np.asarray(prediction))
    return scores_one_exp

def map_labels_to_list(labels, label_list):
    # label_list is a python list with the labels
    # map labels in range(len(label_list)) to the labels in label_list
    # E.g. [0,0,1,1] becomes [0,0,2,2] (if 1 doesnt exist in the data)
    # label gets mapped to label_list[label]
    label_lookup = tf.constant(np.array(label_list))
    return tf.gather(label_lookup, labels)

def build_clf_graph(img_tensor_shape, clf_config):
    graph_classifier = tf.Graph()
    with graph_classifier.as_default():
        # image (batch size = 1)
        x_clf_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # classification of the real source image and the fake target image
        predicted_labels, softmax, age_softmaxs = predict(x_clf_pl, clf_config)
        # scope = tf.get_variable_scope()
        # scope.reuse_variables()

        # map labels in range(len(label_list)) to the labels in label_list
        # E.g. [0,0,1,1] becomes [0,0,2,2] (if 1 doesnt exist in the data)
        predicted_labels_mapped = map_labels_to_list(predicted_labels, clf_config.label_list)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()  # disc loss is scaled negative EM distance
        predictions = {'label': predicted_labels_mapped, 'diag_softmax': softmax, 'age_softmaxs': age_softmaxs}
        return graph_classifier, x_clf_pl, predictions, init, saver


def build_gen_graph(img_tensor_shape, gan_config):
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
    # get savepoint with best crossentropy
    init_checkpoint_path_clf = get_latest_checkpoint_and_log(logdir_clf, 'model_best_xent.ckpt')
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
            source_pred.append(real_pred[i])
        elif field_strength == clf_config.target_field_strength:
            target_indices.append(i)
            target_true_labels.append(labels_test[i])
            target_pred.append(real_pred[i])

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

    # save real images
    target_image_path = os.path.join(image_saving_path, 'target')
    source_image_path = os.path.join(image_saving_path, 'source')
    utils.makefolder(target_image_path)
    utils.makefolder(source_image_path)
    sorted_saving_indices = sorted(image_saving_indices)
    target_saving_indices = [target_indices[index] for index in sorted_saving_indices]
    for target_index in target_saving_indices:
        target_img_name = 'target_img_%.1fT_%d.nii.gz' % (clf_config.target_field_strength, target_index)
        utils.create_and_save_nii(images_test[target_index], os.path.join(target_image_path, target_img_name))
        logging.info(target_img_name + ' saved')

    source_saving_indices = [source_indices[index] for index in sorted_saving_indices]
    for source_index in source_saving_indices:
        source_img_name = 'source_img_%.1fT_%d.nii.gz' % (clf_config.source_field_strength, source_index)
        utils.create_and_save_nii(images_test[source_index], os.path.join(source_image_path, source_img_name))
        logging.info(source_img_name + ' saved')

    logging.info('source and target images saved')

    gan_remainder_batch_size = num_source_images % batch_size

    scores = {}
    for gan_experiment_path in gan_experiment_path_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_path)

        # make sure the experiments all have the same configuration as the classifier
        assert gan_config.source_field_strength == clf_config.source_field_strength
        assert gan_config.target_field_strength == clf_config.target_field_strength
        assert gan_config.image_size == clf_config.image_size
        assert gan_config.target_resolution == clf_config.target_resolution
        assert gan_config.offset == clf_config.offset

        logging.info('\nGAN Experiment (%.1f T to %.1f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_path))
        logging.info(gan_config)
        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        # open the latest GAN savepoint
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator
        graph_generator, generator_img_pl, x_fake_op, init_gan_op, saver_gan = build_gen_graph(img_tensor_shape, gan_config)

        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path = os.path.join(image_saving_path, gan_experiment_path)
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path)

        # make separate graphs for the last batch where the batchsize is smaller
        if clf_remainder_batch_size > 0:
            img_tensor_shape_gan_remainder = [gan_remainder_batch_size, im_s[0], im_s[1], im_s[2], 1]
            # classifier
            graph_clf_rem, image_pl_rem, predictions_clf_op_rem, init_clf_op_rem, saver_clf_rem = build_clf_graph(img_tensor_shape_gan_remainder, clf_config)
            sess_clf_rem = tf.Session(config=config, graph=graph_clf_rem)
            sess_clf_rem.run(init_clf_op_rem)
            saver_clf_rem.restore(sess_clf_rem, init_checkpoint_path_clf)

            # generator
            graph_generator_rem, generator_img_rem_pl, x_fake_op_rem, init_gan_op_rem, saver_gan_rem = build_gen_graph(img_tensor_shape_gan_remainder, gan_config)
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
                generated_img_name = 'generated_img_%.1fT_%d.nii.gz' % (gan_config.target_field_strength, global_index)
                utils.create_and_save_nii(np.squeeze(fake_img[batch_index]), os.path.join(experiment_generate_path, generated_img_name))
                logging.info(generated_img_name + ' saved')
                # save the difference g(xs)-xs
                corresponding_source_img = images_test[global_index]
                difference_image_gs = np.squeeze(fake_img[batch_index]) - corresponding_source_img
                difference_img_name = 'difference_img_%.1fT_%d.nii.gz' % (gan_config.target_field_strength, global_index)
                utils.create_and_save_nii(difference_image_gs, os.path.join(experiment_generate_path, difference_img_name))
                logging.info(difference_img_name + ' saved')

            logging.info('new image batch')
            logging.info('ground truth labels: ' + str(real_label))
            logging.info('predicted labels for generated images: ' + str(clf_prediction_fake['label']))
            # no unexpected labels
            assert all([label in clf_config.label_list for label in clf_prediction_fake['label']])

            batch_beginning_index += current_batch_size
        logging.info('generated prediction for %s: %s' % (gan_experiment_path, str(generated_pred)))
        scores[gan_experiment_path] = evaluate_scores(source_true_labels, generated_pred, score_functions)

    logging.info('source prediction: ' + str(source_pred))
    logging.info('source ground truth: ' + str(source_true_labels))
    logging.info('target prediction: ' + str(target_pred))
    logging.info('target ground truth: ' + str(target_true_labels))

    scores['source'] = evaluate_scores(source_true_labels, source_pred, score_functions)
    scores['target'] = evaluate_scores(target_true_labels, target_pred, score_functions)

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
        images_val, source_images_val_ind, target_images_val_ind = data_utils.get_images_and_fieldstrength_indices(
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
    gan_experiment_list = [
        'bousmalis_bn_dropout_keep0.9_10_noise_all_small_data_0l1_i1',
        'bousmalis_bn_dropout_keep0.9_10_noise_all_small_data_1e5l1_i1',
        'bousmalis_bn_dropout_keep0.9_no_noise_all_small_data_1e5l1_i1',
        'bousmalis_bn_dropout_keep0.9_no_noise_all_small_data_i1',
        'residual_identity_gen_bs2_std_disc_all_small_data_5e5l1_i1',
        'residual_identity_gen_bs2_std_disc_all_small_data_i1',
        'residual_identity_gen_bs20_std_disc_10_noise_all_small_data_1e4l1_bn_i1'
    ]
    clf_experiment_name = 'adni_clf_cropdata_allconv_yesrescale_bs20_all_target15_data_bn_i1'
    clf_log_root = os.path.join(sys_config.log_root, 'adni_clf')
    gan_log_root = os.path.join(sys_config.log_root, 'gan/all_small_images')
    image_saving_path = os.path.join(sys_config.project_root,'data/generated_images/all_data_size_64_80_64_res_1.5_1.5_1.5_lbl_0_2_intrangeone_offset_0_0_-10')
    image_saving_indices = set(range(0, 120, 20))

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





