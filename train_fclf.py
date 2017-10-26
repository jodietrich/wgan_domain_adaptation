# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import logging
import time

import numpy as np
import os.path
import shutil
import tensorflow as tf
from sklearn.metrics import f1_score

import adni_data_loader
import config.system as sys_config
import model_multitask as model_mt
import utils
from batch_generator_list import iterate_minibatches



### EXPERIMENT CONFIG FILE #############################################################

from experiments.fclf import jia_xi_net as exp_config

########################################################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')

def _list_mean(list_of_arrays):

    mean_list = [np.mean(l) for l in list_of_arrays]
    return sum(mean_list)/len(mean_list)


def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

    init_step = 0

    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0

        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # Load data
    data = adni_data_loader.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        label_list=exp_config.diagnosis_list,
        force_overwrite=False
    )

    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train']
    fieldstr_train = data['field_strength_train']
    labels_train = utils.fstr_to_label(fieldstr_train, exp_config.field_strength_list, exp_config.fs_label_list)
    ages_train = data['age_train']

    if exp_config.age_ordinal_regression:
        ages_train = utils.age_to_ordinal_reg_format(ages_train, bins=exp_config.age_bins)
        ordinal_reg_weights = utils.get_ordinal_reg_weights(ages_train)
    else:
        ages_train = utils.age_to_bins(ages_train, bins=exp_config.age_bins)
        ordinal_reg_weights = None

    images_val = data['images_val']
    fieldstr_val = data['field_strength_val']
    labels_val = utils.fstr_to_label(fieldstr_val, exp_config.field_strength_list, exp_config.fs_label_list)
    ages_val = data['age_val']

    if exp_config.age_ordinal_regression:
        ages_val = utils.age_to_ordinal_reg_format(ages_val, bins=exp_config.age_bins)
    else:
        ages_val= utils.age_to_bins(ages_val, bins=exp_config.age_bins)

    if exp_config.use_data_fraction:
        num_images = images_train.shape[0]
        new_last_index = int(float(num_images)*exp_config.use_data_fraction)

        logging.warning('USING ONLY FRACTION OF DATA!')
        logging.warning(' - Number of imgs orig: %d, Number of imgs new: %d' % (num_images, new_last_index))
        images_train = images_train[0:new_last_index,...]
        labels_train = labels_train[0:new_last_index,...]

    logging.info('Data summary:')
    logging.info('TRAINING')
    logging.info(' - Images:')
    logging.info(images_train.shape)
    logging.info(images_train.dtype)
    logging.info(' - Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)
    logging.info('VALIDATiON')
    logging.info(' - Images:')
    logging.info(images_val.shape)
    logging.info(images_val.dtype)
    logging.info(' - Labels:')
    logging.info(labels_val.shape)
    logging.info(labels_val.dtype)

    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        labels_tensor_shape = [exp_config.batch_size]

        if exp_config.age_ordinal_regression:
            ages_tensor_shape = [exp_config.batch_size, len(exp_config.age_bins)]
        else:
            ages_tensor_shape = [exp_config.batch_size]

        images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        diag_placeholder = tf.placeholder(tf.uint8, shape=labels_tensor_shape, name='labels')
        ages_placeholder = tf.placeholder(tf.uint8, shape=ages_tensor_shape, name='ages')

        learning_rate_placeholder = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training_time_placeholder = tf.placeholder(tf.bool, shape=[], name='training_time')

        tf.summary.scalar('learning_rate', learning_rate_placeholder)

        # Build a Graph that computes predictions from the inference model.
        diag_logits, ages_logits = exp_config.model_handle(images_placeholder,
                                                           nlabels=exp_config.nlabels,
                                                           training=training_time_placeholder,
                                                           n_age_thresholds=len(exp_config.age_bins),
                                                           bn_momentum=exp_config.bn_momentum)

        # Add to the Graph the Ops for loss calculation.

        [loss, diag_loss, age_loss, weights_norm] = model_mt.loss(diag_logits,
                                                                  ages_logits,
                                                                  diag_placeholder,
                                                                  ages_placeholder,
                                                                  nlabels=exp_config.nlabels,
                                                                  weight_decay=exp_config.weight_decay,
                                                                  diag_weight=exp_config.diag_weight,
                                                                  age_weight=exp_config.age_weight,
                                                                  use_ordinal_reg=exp_config.age_ordinal_regression,
                                                                  ordinal_reg_weights=ordinal_reg_weights)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('diag_loss', diag_loss)
        tf.summary.scalar('age_loss', age_loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        if exp_config.momentum is not None:
            optimiser = exp_config.optimizer_handle(learning_rate=learning_rate_placeholder,
                                                    momentum=exp_config.momentum)
        else:
            optimiser = exp_config.optimizer_handle(learning_rate=learning_rate_placeholder)

        # create a copy of all trainable variables with `0` as initial values
        t_vars = tf.global_variables() #tf.trainable_variables()
        accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in t_vars]

        # create a op to initialize all accums vars
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

        # compute gradients for a batch
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            batch_grads_vars = optimiser.compute_gradients(loss, t_vars)

        # collect the batch gradient into accumulated vars

        accum_ops = [accum_tvar.assign_add(batch_grad_var[0]) for accum_tvar, batch_grad_var in zip(accum_tvars, batch_grads_vars)]

        accum_normaliser_pl = tf.placeholder(dtype=tf.float32, name='accum_normaliser')
        accum_mean_op = [accum_tvar.assign(tf.divide(accum_tvar, accum_normaliser_pl)) for accum_tvar in accum_tvars]

        # apply accums gradients
        with tf.control_dependencies(update_ops):
            train_op = optimiser.apply_gradients(
                [(accum_tvar, batch_grad_var[1]) for accum_tvar, batch_grad_var in zip(accum_tvars, batch_grads_vars)]
            )

        eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs = model_mt.evaluation(diag_logits, ages_logits,
                                                                                         diag_placeholder,
                                                                                         ages_placeholder,
                                                                                         images_placeholder,
                                                                                         diag_weight=exp_config.diag_weight,
                                                                                         age_weight=exp_config.age_weight,
                                                                                         nlabels=exp_config.nlabels,
                                                                                         use_ordinal_reg=exp_config.age_ordinal_regression)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=3)
        saver_best_diag_f1 = tf.train.Saver(max_to_keep=2)
        saver_best_ages_f1 = tf.train.Saver(max_to_keep=2)
        saver_best_xent = tf.train.Saver(max_to_keep=2)

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # with tf.name_scope('monitoring'):

        val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error_diag')
        val_error_summary = tf.summary.scalar('validation_loss', val_error_)

        val_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='val_diag_f1')
        val_f1_diag_summary = tf.summary.scalar('validation_diag_f1', val_diag_f1_score_)

        val_ages_f1_score_ = tf.placeholder(tf.float32, shape=[], name='val_ages_f1')
        val_f1_ages_summary = tf.summary.scalar('validation_ages_f1', val_ages_f1_score_)

        val_summary = tf.summary.merge([val_error_summary, val_f1_diag_summary, val_f1_ages_summary])

        train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error_diag')
        train_error_summary = tf.summary.scalar('training_loss', train_error_)

        train_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='train_diag_f1')
        train_diag_f1_summary = tf.summary.scalar('training_diag_f1', train_diag_f1_score_)

        train_ages_f1_score_ = tf.placeholder(tf.float32, shape=[], name='train_ages_f1')
        train_f1_ages_summary = tf.summary.scalar('training_ages_f1', train_ages_f1_score_)

        train_summary = tf.summary.merge([train_error_summary, train_diag_f1_summary, train_f1_ages_summary])

        # Run the Op to initialize the variables.
        sess.run(init)

        if continue_run:
            # Restore session
            saver.restore(sess, init_checkpoint_path)

        step = init_step
        curr_lr = exp_config.learning_rate

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_diag_f1_score = 0
        best_ages_f1_score = 0

        # acum_manual = 0  #np.zeros((2,3,3,3,1,32))

        for epoch in range(exp_config.max_epochs):

            logging.info('EPOCH %d' % epoch)
            sess.run(zero_ops)
            accum_counter = 0

            for batch in iterate_minibatches(images_train,
                                             [labels_train, ages_train],
                                             batch_size=exp_config.batch_size,
                                             augmentation_function=exp_config.augmentation_function,
                                             exp_config=exp_config):


                if exp_config.warmup_training:
                    if step < 50:
                        curr_lr = exp_config.learning_rate / 10.0
                    elif step == 50:
                        curr_lr = exp_config.learning_rate

                start_time = time.time()

                # get a batch
                x, [y, a] = batch

                # TEMPORARY HACK (to avoid incomplete batches)
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue


                # Run accumulation
                feed_dict = {
                    images_placeholder: x,
                    diag_placeholder: y,
                    ages_placeholder: a,
                    learning_rate_placeholder: curr_lr,
                    training_time_placeholder: True
                }

                _, loss_value = sess.run([accum_ops, loss], feed_dict=feed_dict)

                accum_counter += 1

                if accum_counter == exp_config.n_accum_batches:

                    # Average gradient over batches
                    sess.run(accum_mean_op, feed_dict={accum_normaliser_pl: float(exp_config.n_accum_batches)})
                    sess.run(train_op, feed_dict={learning_rate_placeholder: curr_lr, training_time_placeholder: True})

                    # Reset all counters etc.
                    sess.run(zero_ops)
                    accum_counter = 0

                    duration = time.time() - start_time

                    # Write the summaries and print an overview fairly often.
                    if step % 10 == 0:
                        # Print status to stdout.


                        logging.info('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                        # Update the events file.

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    if (step + 1) % exp_config.train_eval_frequency == 0:

                        # Evaluate against the training set
                        logging.info('Training Data Eval:')
                        [train_loss, train_diag_f1, train_ages_f1] = do_eval(sess,
                                                                             eval_diag_loss,
                                                                             eval_ages_loss,
                                                                             pred_labels,
                                                                             ages_softmaxs,
                                                                             images_placeholder,
                                                                             diag_placeholder,
                                                                             ages_placeholder,
                                                                             training_time_placeholder,
                                                                             images_train,
                                                                             [labels_train, ages_train],
                                                                             batch_size=exp_config.batch_size,
                                                                             do_ordinal_reg=exp_config.age_ordinal_regression)


                        train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                               train_diag_f1_score_: train_diag_f1,
                                                                               train_ages_f1_score_: train_ages_f1}
                                                     )
                        summary_writer.add_summary(train_summary_msg, step)

                        loss_history.append(train_loss)
                        if len(loss_history) > 5:
                            loss_history.pop(0)
                            loss_gradient = (loss_history[-5] - loss_history[-1]) / 2

                        logging.info('loss gradient is currently %f' % loss_gradient)

                        if exp_config.schedule_lr and loss_gradient < exp_config.schedule_gradient_threshold:
                            logging.warning('Reducing learning rate!')
                            curr_lr /= 10.0
                            logging.info('Learning rate changed to: %f' % curr_lr)

                            # reset loss history to give the optimisation some time to start decreasing again
                            loss_gradient = np.inf
                            loss_history = []

                        if train_loss <= last_train:  # best_train:
                            logging.info('Decrease in training error!')
                        else:
                            logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                        last_train = train_loss

                    # Save a checkpoint and evaluate the model periodically.
                    if (step + 1) % exp_config.val_eval_frequency == 0:

                        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=step)

                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')

                        [val_loss, val_diag_f1, val_ages_f1] = do_eval(sess,
                                                                       eval_diag_loss,
                                                                       eval_ages_loss,
                                                                       pred_labels,
                                                                       ages_softmaxs,
                                                                       images_placeholder,
                                                                       diag_placeholder,
                                                                       ages_placeholder,
                                                                       training_time_placeholder,
                                                                       images_val,
                                                                       [labels_val, ages_val],
                                                                       batch_size=exp_config.batch_size,
                                                                       do_ordinal_reg=exp_config.age_ordinal_regression)


                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss,
                                                                           val_diag_f1_score_: val_diag_f1,
                                                                           val_ages_f1_score_: val_ages_f1}
                        )
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_diag_f1 >= best_diag_f1_score:
                            best_diag_f1_score = val_diag_f1
                            best_file = os.path.join(log_dir, 'model_best_diag_f1.ckpt')
                            saver_best_diag_f1.save(sess, best_file, global_step=step)
                            logging.info('Found new best DIAGNOSIS F1 score on validation set! - %f -  Saving model_best_diag_f1.ckpt' % val_diag_f1)

                        if val_ages_f1 >= best_ages_f1_score:
                            best_ages_f1_score = val_ages_f1
                            best_file = os.path.join(log_dir, 'model_best_ages_f1.ckpt')
                            saver_best_ages_f1.save(sess, best_file, global_step=step)
                            logging.info('Found new best AGES F1 score on validation set! - %f -  Saving model_best_ages_f1.ckpt' % val_ages_f1)

                        if val_loss <= best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'model_best_xent.ckpt')
                            saver_best_xent.save(sess, best_file, global_step=step)
                            logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

                    step += 1

        sess.close()


def do_eval(sess,
            eval_diag_loss,
            eval_ages_loss,
            pred_labels,
            ages_softmaxs,
            images_placeholder,
            diag_labels_placeholder,
            ages_placeholder,
            training_time_placeholder,
            images,
            labels_list,
            batch_size,
            do_ordinal_reg):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels_list: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    diag_loss_ii = 0
    ages_loss_ii = 0
    num_batches = 0
    predictions_diag = []
    predictions_diag_gt = []
    predictions_ages = []
    predictions_ages_gt = []

    for batch in iterate_minibatches(images,
                                     labels_list,
                                     batch_size=batch_size,
                                     augmentation_function=None,
                                     exp_config=exp_config):  # No aug in evaluation
    # As before you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
    # but at the risk of not catching exceptions

        x, [y, a] = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      diag_labels_placeholder: y,
                      ages_placeholder: a,
                      training_time_placeholder: False}

        c_d_loss, c_a_loss, c_d_preds, c_a_softmaxs = sess.run([eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs], feed_dict=feed_dict)

        # This converts the labels back into the original format. I.e. [0,1,1,0] will become [0,2,2,0] again if
        # 1 didn't exist in the dataset.
        c_d_preds = [exp_config.fs_label_list[pp] for pp in c_d_preds]
        y_gts = [exp_config.fs_label_list[pp] for pp in y]

        diag_loss_ii += c_d_loss
        ages_loss_ii += c_a_loss
        num_batches += 1
        predictions_diag += c_d_preds
        predictions_diag_gt += y_gts

        if do_ordinal_reg:

            c_a_preds = np.asarray(c_a_softmaxs)
            c_a_preds = np.transpose(c_a_preds, (1, 0, 2))
            c_a_preds = c_a_preds[:, :, 1]
            c_a_preds = np.uint8(c_a_preds + 0.5)

            predictions_ages += list(utils.ordinal_regression_to_bin(c_a_preds))
            predictions_ages_gt += list(utils.ordinal_regression_to_bin(a))

        else:

            c_a_preds = np.argmax(c_a_softmaxs, axis=-1)

            predictions_ages += list(c_a_preds)
            predictions_ages_gt += list(a)


    avg_loss = (diag_loss_ii / num_batches) + (ages_loss_ii / num_batches)


    f1_diag_score = f1_score(np.asarray(predictions_diag_gt), np.asarray(predictions_diag), average='micro')  # micro is overall, macro doesn't take class imbalance into account
    # f1_ages_score = f1_score(np.asarray(predictions_ages_gt), np.asarray(predictions_ages), average='micro')  # micro is overall, macro doesn't take class imbalance into account

    f1_ages_score = np.mean(np.abs(np.asarray(predictions_ages, dtype=np.int32) - np.asarray(predictions_ages_gt,  dtype=np.int32)))

    logging.info('  Average loss: %0.04f, diag f1_score: %0.04f, age f1_score: %0.04f' % (avg_loss, f1_diag_score, f1_ages_score))

    return avg_loss, f1_diag_score, f1_ages_score



def main():

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    main()
