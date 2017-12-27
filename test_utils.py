import logging

import numpy as np
import tensorflow as tf
from collections import OrderedDict

import utils
from clf_model_multitask import predict


def get_latest_checkpoint_and_log(logdir, filename):
    init_checkpoint_path = utils.get_latest_model_checkpoint_path(logdir, filename)
    logging.info('Checkpoint path: %s' % init_checkpoint_path)
    last_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step)
    return init_checkpoint_path


def evaluate_scores(true_labels, prediction, measures_dict):
    scores_one_exp = OrderedDict()
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


def build_clf_graph(img_tensor_shape, clf_config, joint=False):
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
    # noise_shape
    generator = gan_config.generator
    graph_generator = tf.Graph()
    with graph_generator.as_default():
        # source image (batch size = 1)
        xs_pl = tf.placeholder(tf.float32, img_tensor_shape, name='xs_pl')

        if gan_config.use_generator_input_noise:
            noise_shape = gan_config.generator_input_noise_shape.copy()
            # adjust batch size
            noise_shape[0] = img_tensor_shape[0]
            noise_in_gen_pl = tf.random_uniform(shape=noise_shape, minval=-1, maxval=1)
        else:
            noise_in_gen_pl = None

        # generated fake image batch
        xf = generator(xs=xs_pl, z_noise=noise_in_gen_pl, training=False)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()
        return graph_generator, xs_pl, xf, init, saver