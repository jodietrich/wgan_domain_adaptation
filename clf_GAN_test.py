

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
        scores_one_exp[measure_name] = measure(y_true = np.asarray(true_labels), y_pred = np.asarray(prediction), average='micro')  # micro is overall, macro doesn't take class imbalance into account
    return scores_one_exp


def build_clf_graph(img_tensor_shape, clf_config):
    graph_classifier = tf.Graph()
    with graph_classifier.as_default():
        # image (batch size = 1)
        x_clf_pl = tf.placeholder(tf.float32, img_tensor_shape, name='z')

        # classification of the real source image and the fake target image
        predicted_labels, softmax, age_softmaxs = predict(x_clf_pl, clf_config)
        # scope = tf.get_variable_scope()
        # scope.reuse_variables()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()  # disc loss is scaled negative EM distance
        predictions = {'label': predicted_labels, 'diag_softmax': softmax, 'age_softmaxs': age_softmaxs}
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


def generate_and_evaluate_ad_classification(gan_experiment_list, clf_experiment_name, num_images_to_save=0, image_saving_path=None, max_batch_size = np.inf):
    """

    :param gan_experiment_list: list of GAN experiment names to be evaluated. They must all have the same image settings and source/target field strengths as the classifier
    :param clf_experiment_name: AD classifier used
    :param verbose: boolean. log all image classifications
    :param num_images_to_save: how many images to save as nii files. 0 if none should be saved
    :param image_saving_path: where to save the images. They are saved in subfolders for each experiment
    :return:
    """

    clf_config, logdir_clf = utils.load_log_exp_config(clf_experiment_name)

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

    num_images = images_test.shape[0]
    logging.info('there are %d test images')

    im_s = clf_config.image_size
    batch_size = min(clf_config.batch_size, std_params.batch_size, max_batch_size)
    img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]
    clf_remainder_batch_size = images_test.shape[0] % batch_size

    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

    # open field strength classifier save file from the selected experiment
    logging.info("loading Alzheimer's disease classifier")
    graph_clf, image_pl, predictions_clf_op, init_clf_op, saver_clf = build_clf_graph(img_tensor_shape, clf_config)
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
    real_pred = []
    for batch in iterate_minibatches(images_test,
                                     [labels_test, ages_test],
                                     batch_size=batch_size,
                                     exp_config=clf_config,
                                     shuffle_data=False,
                                     skip_remainder=False):
        # ignore the labels because data are in order, which means the label list in data can be used
        image_batch = batch[0]

        current_batch_size = image_batch.shape[0]
        if current_batch_size < batch_size:
            clf_prediction_real = sess_clf_rem.run(predictions_clf_op_rem, feed_dict={image_pl_rem: image_batch})
        else:
            clf_prediction_real = sess_clf.run(predictions_clf_op, feed_dict={image_pl: image_batch})

        real_pred = real_pred + clf_prediction_real['label']

    source_indices = set()
    target_indices = set()
    source_true_labels = []
    source_pred = []
    target_true_labels = []
    target_pred = []
    for i, field_strength in enumerate(data['field_strength_test']):
        if field_strength == clf_config.source_field_strength:
            source_indices.add(i)
            source_true_labels.append(labels_test[i])
            source_pred.append(real_pred[i])
        elif field_strength == clf_config.target_field_strength:
            target_indices.add(i)
            target_true_labels.append(labels_test[i])
            target_pred.append(real_pred[i])

    num_source_images = len(source_indices)
    gan_remainder_batch_size = num_source_images % batch_size

    scores = {}
    num_images_already_saved = 0
    for gan_experiment_name in gan_experiment_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_name)
        logging.info('\nGAN Experiment (%f T to %f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))

        # open GAN save file from the selected experiment
        logging.info('loading GAN')
        init_checkpoint_path_gan = get_latest_checkpoint_and_log(logdir_gan, 'model.ckpt')

        # build a separate graph for the generator
        graph_generator, generator_img_pl, x_fake_op, init_gan_op, saver_gan = build_gen_graph(img_tensor_shape, gan_config)

        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path = os.path.join(image_saving_path, gan_experiment_name)
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



        generated_pred = []
        # loops through all images from the source domain
        for batch in iterate_minibatches(images_test,
                                     [labels_test, ages_test],
                                     batch_size=batch_size,
                                     exp_config=clf_config,
                                     selection_indices=list(source_indices),
                                     shuffle_data=False,
                                     skip_remainder=False):
            # ignore the labels because data are in order, which means the label list in data can be used
            image_batch = batch[0]

            current_batch_size = image_batch.shape[0]
            if current_batch_size < batch_size:
                fake_img = sess_gan_rem.run(x_fake_op_rem, feed_dict={generator_img_rem_pl: image_batch})
                # classify fake image
                clf_prediction_fake = sess_clf_rem.run(predictions_clf_op_rem, feed_dict={image_pl_rem: fake_img})
            else:
                fake_img = sess_gan.run(x_fake_op, feed_dict={generator_img_pl: image_batch})
                # classify fake image
                clf_prediction_fake = sess_clf.run(predictions_clf_op, feed_dict={image_pl: fake_img})

            generated_pred = generated_pred + clf_prediction_fake['label']

            # save images
            for real_image, generated_image in itertools.izip(image_batch, fake_img):
                if num_images_already_saved < num_images_to_save:
                    source_img_name = 'source_img_%.1fT_%d.nii.gz' % (gan_config.source_field_strength, num_images_already_saved)
                    generated_img_name = 'generated_img_%.1fT_%d.nii.gz' % (gan_config.target_field_strength, num_images_already_saved)
                    utils.create_and_save_nii(np.squeeze(real_image), os.path.join(experiment_generate_path, source_img_name))
                    utils.create_and_save_nii(np.squeeze(generated_image), os.path.join(experiment_generate_path, generated_img_name))
                    logging.info('images saved')
                    num_images_already_saved += 1
                else:
                    break

        # separate true labels and predicted labels for real images into source and target domain


        measures_dict = {'f1': f1_score, 'recall': recall_score, 'precision': precision_score}
        scores[gan_experiment_name] = evaluate_scores(source_true_labels, generated_pred, measures_dict)

    scores['source'] = evaluate_scores(source_true_labels, source_pred)
    scores['target'] = evaluate_scores(target_true_labels, target_pred)

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
        'bousmalis_bn_dropout_keep0.9_no_noise_all_small_data_i1',
        'residual_identity_gen_bs2_std_disc_all_small_data_i1'
    ]
    fclf_experiment_name = 'adni_clf_cropdata_allconv_yesrescale_bs20_all_target15_data_bn_i1'
    image_saving_path = os.path.join(sys_config.project_root,'data/generated_images')

    # import config file for field strength classifier
    logging.info('Classifier used: ' + fclf_experiment_name)

    clf_scores = generate_and_evaluate_ad_classification(gan_experiment_list, fclf_experiment_name, num_images_to_save=10, image_saving_path=image_saving_path, max_batch_size=np.inf)
    logging.info(clf_scores)
    gen_f1_score = lambda exp_name: clf_scores[exp_name]['f1']

    scores_string = utils.string_dict_in_order(clf_scores, key=gen_f1_score)
    logging.info('FINAL SUMMARY:\nordered by f1 score\n' + scores_string)

    # gives the name of the experiment with the best f1 score on the generated images
    best_experiment = max(clf_scores, key=gen_f1_score)
    best_score = gen_f1_score(best_experiment)
    logging.info('The best experiment was %s with f1 score %f for the generated images' % (best_experiment, best_score))





