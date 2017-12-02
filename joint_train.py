# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import shutil
from sklearn.metrics import f1_score

import config.system as sys_config
import gan_model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader_all
import data_utils
from batch_generator_list import iterate_minibatches_endlessly, iterate_minibatches
import clf_model_multitask as clf_model_mt
import joint_model



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

#######################################################################
from experiments.joint import clf_allconv_gan_bousmalis as exp_config
#######################################################################

log_dir = os.path.join(sys_config.log_root, exp_config.log_folder, exp_config.experiment_name)

try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')

def run_training(continue_run):

    logging.info('===== RUNNING EXPERIMENT ========')
    logging.info(exp_config.experiment_name)
    logging.info('=================================')

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

    # import data
    data = adni_data_loader_all.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        label_list = exp_config.label_list,
        offset=exp_config.offset,
        rescale_to_one=exp_config.rescale_to_one,
        force_overwrite=False
    )

    # extract images and indices of source/target images for the training and validation set
    images_train, source_images_train_ind, target_images_train_ind,\
    images_val, source_images_val_ind, target_images_val_ind = data_utils.get_images_and_fieldstrength_indices(
        data, exp_config.source_field_strength, exp_config.target_field_strength)

    # get labels
    # the following are HDF5 datasets, not numpy arrays
    labels_train = data['diagnosis_train']
    ages_train = data['age_train']
    labels_val = data['diagnosis_val']
    ages_val = data['age_val']

    if exp_config.age_ordinal_regression:
        ages_train = utils.age_to_ordinal_reg_format(ages_train, bins=exp_config.age_bins)
        ordinal_reg_weights = utils.get_ordinal_reg_weights(ages_train)
    else:
        ages_train = utils.age_to_bins(ages_train, bins=exp_config.age_bins)
        ordinal_reg_weights = None

    if exp_config.age_ordinal_regression:
        ages_val = utils.age_to_ordinal_reg_format(ages_val, bins=exp_config.age_bins)
    else:
        ages_val= utils.age_to_bins(ages_val, bins=exp_config.age_bins)

    generator = exp_config.generator
    discriminator = exp_config.discriminator
    augmentation_function = exp_config.augmentation_function if exp_config.use_augmentation else None

    s_sampler_train = iterate_minibatches_endlessly(images_train,
                                                    batch_size=2*exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    labels_list=[labels_train, ages_train],
                                                    selection_indices=source_images_train_ind,
                                                    augmentation_function=augmentation_function)

    t_sampler_train = iterate_minibatches_endlessly(images_train,
                                                    batch_size=exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    labels_list=[labels_train, ages_train],
                                                    selection_indices=target_images_train_ind,
                                                    augmentation_function=augmentation_function)


    with tf.Graph().as_default():

        training_time_placeholder = tf.placeholder(tf.bool, shape=[], name='training_time')

        # GAN

        # input noise for generator
        if exp_config.use_generator_input_noise:
            noise_in_gen_pl = tf.random_uniform(shape=exp_config.generator_input_noise_shape, minval=-1, maxval=1)
        else:
            noise_in_gen_pl = None

        # target image batch
        xt_pl = tf.placeholder(tf.float32, image_tensor_shape(exp_config.batch_size), name='x_target')

        # source image batch
        xs_pl, diag_s_pl, ages_s_pl = placeholders_clf(2*exp_config.batch_size, 'source')
        # split source batch into 1 to be translated to xf and 2 for the classifier
        # for the discriminator train op half 2 of the batch is not used
        xs1_pl, xs2_pl = tf.split(xs_pl, 2, axis=0)

        # generated fake image batch
        xf_pl = generator(xs1_pl, noise_in_gen_pl, training_time_placeholder)

        # difference between generated and source images
        diff_img_pl = xf_pl - xs1_pl

        # visualize the images by showing one slice of them in the z direction
        tf.summary.image('sample_outputs', tf_utils.put_kernels_on_grid3d(xf_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_xt', tf_utils.put_kernels_on_grid3d(xt_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_xs', tf_utils.put_kernels_on_grid3d(xs1_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_difference_xf-xs', tf_utils.put_kernels_on_grid3d(diff_img_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='centered',
                                                                          cutoff_abs=exp_config.diff_threshold))

        # output of the discriminator for real image
        d_pl = discriminator(xt_pl, training_time_placeholder, scope_reuse=False)

        # output of the discriminator for fake image
        d_pl_ = discriminator(xf_pl, training_time_placeholder, scope_reuse=True)

        d_hat = None
        x_hat = None
        if exp_config.improved_training:

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * xt_pl + (1 - epsilon) * xf_pl
            d_hat = discriminator(x_hat, training_time_placeholder, scope_reuse=True)

        dist_l1 = tf.reduce_mean(tf.abs(diff_img_pl))

        learning_rate_gan_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        learning_rate_clf_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        if exp_config.momentum is not None:
            optimizer_handle = lambda learning_rate: exp_config.optimizer_handle(learning_rate=learning_rate,
                                                    momentum=exp_config.momentum)
        else:
            optimizer_handle = lambda learning_rate: exp_config.optimizer_handle(learning_rate=learning_rate)

        # Build the operation for clipping the discriminator weights
        d_clip_op = gan_model.clip_op()

        # Put L1 distance of generated image and original image on summary
        dist_l1_summary_op = tf.summary.scalar('L1_distance_to_source_img', dist_l1)

        # Classifier ----------------------------------------------------------------------------------------
        # for training usually false so xt and xf get concatenated as classifier input, otherwise
        directly_feed_clf_pl = tf.placeholder(tf.bool, shape=[], name='direct_classifier_feeding')

        # conditionally assign either a concatenation of the generated dataset and the source data
        # cond to avoid having to specify not needed placeholders in the feed dict
        images_clf, diag_clf, ages_clf = tf.cond(
            directly_feed_clf_pl,
            lambda: placeholders_clf(2*exp_config.batch_size, 'direct_clf'),
            lambda: concatenate_clf_input([xf_pl, xs2_pl], diag_s_pl, ages_s_pl, scope_name = 'fs_concat')
        )

        tf.summary.scalar('learning_rate_gan', learning_rate_gan_pl)
        tf.summary.scalar('learning_rate_clf', learning_rate_clf_pl)

        # Build a Graph that computes predictions from the inference model.
        diag_logits_train, ages_logits_train = exp_config.clf_model_handle(images_clf,
                                                                           nlabels=exp_config.nlabels,
                                                                           training=training_time_placeholder,
                                                                           n_age_thresholds=len(exp_config.age_bins),
                                                                           bn_momentum=exp_config.bn_momentum)

        # Add to the Graph the Ops for loss calculation.

        [classifier_loss, diag_loss, age_loss, weights_norm_clf] = clf_model_mt.loss(diag_logits_train,
                                                                                 ages_logits_train,
                                                                                 diag_clf,
                                                                                 ages_clf,
                                                                                 nlabels=exp_config.nlabels,
                                                                                 weight_decay=exp_config.weight_decay,
                                                                                 diag_weight=exp_config.diag_weight,
                                                                                 age_weight=exp_config.age_weight,
                                                                                 use_ordinal_reg=exp_config.age_ordinal_regression,
                                                                                 ordinal_reg_weights=ordinal_reg_weights)

        # nr means no regularization, meaning the loss without the regularization term
        train_ops_dict, losses_gan_dict = joint_model.training_ops(d_pl, d_pl_,
                                                             classifier_loss,
                                                             optimizer_handle=optimizer_handle,
                                                             learning_rate_gan=learning_rate_gan_pl,
                                                             learning_rate_clf=learning_rate_clf_pl,
                                                             l1_img_dist=dist_l1,
                                                             gan_loss_weight=exp_config.gan_loss_weight,
                                                             task_loss_weight=exp_config.task_loss_weight,
                                                             w_reg_img_dist_l1=exp_config.w_reg_img_dist_l1,
                                                             w_reg_gen_l1=exp_config.w_reg_gen_l1,
                                                             w_reg_disc_l1=exp_config.w_reg_disc_l1,
                                                             w_reg_gen_l2=exp_config.w_reg_gen_l2,
                                                             w_reg_disc_l2=exp_config.w_reg_disc_l2,
                                                             d_hat=d_hat, x_hat=x_hat, scale=exp_config.scale)


        tf.summary.scalar('classifier loss', classifier_loss)
        tf.summary.scalar('diag_loss', diag_loss)
        tf.summary.scalar('age_loss', age_loss)
        tf.summary.scalar('weights_norm_term_classifier', weights_norm_clf)
        tf.summary.scalar('generator loss joint', losses_gan_dict['gen']['joint'])
        tf.summary.scalar('discriminator loss joint', losses_gan_dict['disc']['joint'])

        eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs = clf_model_mt.evaluation(diag_logits_train, ages_logits_train,
                                                                                             diag_clf,
                                                                                             ages_clf,
                                                                                             images_clf,
                                                                                             diag_weight=exp_config.diag_weight,
                                                                                             age_weight=exp_config.age_weight,
                                                                                             nlabels=exp_config.nlabels,
                                                                                             use_ordinal_reg=exp_config.age_ordinal_regression)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()


        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver_latest = tf.train.Saver(max_to_keep=2)
        saver_best_disc = tf.train.Saver(max_to_keep=2)  # disc loss is scaled negative EM distance
        saver_best_diag_f1 = tf.train.Saver(max_to_keep=2)
        saver_best_ages_f1 = tf.train.Saver(max_to_keep=2)
        saver_best_xent = tf.train.Saver(max_to_keep=2)

        # validation summaries gan
        val_disc_loss_pl = tf.placeholder(tf.float32, shape=[], name='disc_val_loss')
        disc_val_summary_op = tf.summary.scalar('validation_discriminator_loss', val_disc_loss_pl)

        val_gen_loss_pl = tf.placeholder(tf.float32, shape=[], name='gen_val_loss')
        gen_val_summary_op = tf.summary.scalar('validation_generator_loss', val_gen_loss_pl)

        val_summary_gan = tf.summary.merge([disc_val_summary_op, gen_val_summary_op])

        # Classifier summary
        val_error_clf_ = tf.placeholder(tf.float32, shape=[], name='val_error_diag')
        val_error_summary = tf.summary.scalar('classifier_validation_loss', val_error_clf_)

        val_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='val_diag_f1')
        val_f1_diag_summary = tf.summary.scalar('validation_diag_f1', val_diag_f1_score_)

        val_ages_f1_score_ = tf.placeholder(tf.float32, shape=[], name='val_ages_f1')
        val_f1_ages_summary = tf.summary.scalar('validation_ages_f1', val_ages_f1_score_)

        val_summary_clf = tf.summary.merge([val_error_summary, val_f1_diag_summary, val_f1_ages_summary])
        val_summary = tf.summary.merge([val_summary_clf, val_summary_gan])

        train_error_clf_ = tf.placeholder(tf.float32, shape=[], name='train_error_diag')
        train_error_clf_summary = tf.summary.scalar('classifier_training_loss', train_error_clf_)

        train_diag_f1_score_ = tf.placeholder(tf.float32, shape=[], name='train_diag_f1')
        train_diag_f1_summary = tf.summary.scalar('training_diag_f1', train_diag_f1_score_)

        train_ages_f1_score_ = tf.placeholder(tf.float32, shape=[], name='train_ages_f1')
        train_f1_ages_summary = tf.summary.scalar('training_ages_f1', train_ages_f1_score_)

        train_summary = tf.summary.merge([train_error_clf_summary, train_diag_f1_summary, train_f1_ages_summary])

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.graph.finalize()

        # Run the Op to initialize the variables.
        sess.run(init)

        if continue_run:
            # Restore session
            saver_latest.restore(sess, init_checkpoint_path)

        curr_lr_gan = exp_config.learning_rate_gan
        curr_lr_clf = exp_config.learning_rate_clf

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_diag_f1_score = 0
        best_ages_f1_score = 0
        # initialize value of lowest (i. e. best) discriminator loss
        best_d_loss = np.inf

        for step in range(init_step, exp_config.max_steps):

            start_time = time.time()

            # discriminator and classifier (task) training iterations
            d_iters = 5
            t_iters = 1
            if step % 500 == 0 or step < 25:
                d_iters = 100
            for iteration in range(max(d_iters, t_iters)):

                x_t, [diag_t, age_t] = next(t_sampler_train)
                x_s, [diag_s, age_s] = next(s_sampler_train)

                feed_dict_dc = {xs_pl: x_s,
                                xt_pl: x_t,
                                learning_rate_gan_pl: curr_lr_gan,
                                learning_rate_clf_pl: curr_lr_clf,
                                diag_s_pl: diag_s,
                                ages_s_pl: age_s,
                                training_time_placeholder: True,
                                directly_feed_clf_pl: False}
                train_ops_list_dc = []
                if iteration < t_iters:
                    # train classifier
                    train_ops_list_dc.append(train_ops_dict['clf'])

                if iteration < d_iters:
                    # train discriminator
                    train_ops_list_dc.append(train_ops_dict['disc'])

                if not exp_config.improved_training:
                    sess.run(d_clip_op)

            elapsed_time = time.time() - start_time

            # train generator
            x_t, [diag_t, age_t] = next(t_sampler_train)
            x_s, [diag_s, age_s] = next(s_sampler_train)
            sess.run(train_ops_dict['gen'],
                     feed_dict={xs_pl: x_s,
                                xt_pl: x_t,
                                learning_rate_gan_pl: curr_lr_gan,
                                learning_rate_clf_pl: curr_lr_clf,
                                diag_s_pl: diag_s,
                                ages_s_pl: age_s,
                                training_time_placeholder: True,
                                directly_feed_clf_pl: False
                                })

            if step % exp_config.update_tensorboard_frequency == 0:
                x_t, [diag_t, age_t] = next(t_sampler_train)
                x_s, [diag_s, age_s] = next(s_sampler_train)

                feed_dict_summary={xs_pl: x_s,
                                    xt_pl: x_t,
                                    learning_rate_gan_pl: curr_lr_gan,
                                    learning_rate_clf_pl: curr_lr_clf,
                                    diag_s_pl: diag_s,
                                    ages_s_pl: age_s,
                                    training_time_placeholder: True,
                                    directly_feed_clf_pl: False
                                    }

                c_loss_one_batch, gan_losses_one_batch_dict, summary_str = sess.run(
                        [classifier_loss, losses_gan_dict, summary], feed_dict=feed_dict_summary)

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                logging.info("[Step: %d], classifier_loss: %g, GAN losses: %s" % (step, c_loss_one_batch, str(gan_losses_one_batch_dict)))
                logging.info(" - elapsed time for one step: %f secs" % elapsed_time)

            if (step + 1) % exp_config.train_eval_frequency == 0:

                # Evaluate against the training set
                logging.info('Training data eval for classifier (target domain):')
                [train_loss, train_diag_f1, train_ages_f1] = do_eval_classifier(sess, eval_diag_loss,
                                                                                eval_ages_loss,
                                                                                pred_labels,
                                                                                ages_softmaxs,
                                                                                xs_pl,
                                                                                diag_s_pl,
                                                                                ages_s_pl,
                                                                                training_time_placeholder,
                                                                                directly_feed_clf_pl,
                                                                                images_train,
                                                                                [labels_train, ages_train],
                                                                                clf_batch_size=2 * exp_config.batch_size,
                                                                                do_ordinal_reg=exp_config.age_ordinal_regression,
                                                                                selection_indices=source_images_train_ind)

                train_summary_msg = sess.run(train_summary, feed_dict={train_error_clf_: train_loss,
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
                    logging.warning('Reducing learning rate of the classifier!')
                    curr_lr_clf /= 10.0
                    logging.info('Learning rate of the classifier changed to: %f' % curr_lr_clf)

                    # reset loss history to give the optimisation some time to start decreasing again
                    loss_gradient = np.inf
                    loss_history = []

                if train_loss <= last_train:  # best_train:
                    logging.info('Decrease in training error!')
                else:
                    logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                last_train = train_loss


            if (step + 1) % exp_config.validation_frequency == 0:

                # evaluate gan losses
                g_loss_val_avg, d_loss_val_avg = do_eval_gan(sess=sess,
                                                             losses=[losses_gan_dict['gen']['nr'], losses_gan_dict['disc']['nr']],
                                                             images_s_pl=xs_pl,
                                                             images_t_pl=xt_pl,
                                                             training_time_placeholder=training_time_placeholder,
                                                             images=images_val,
                                                             source_images_ind=source_images_val_ind,
                                                             target_images_ind=target_images_val_ind)

                # evaluate classifier losses
                [val_loss, val_diag_f1, val_ages_f1] = do_eval_classifier(sess,
                                                                          eval_diag_loss,
                                                                          eval_ages_loss,
                                                                          pred_labels,
                                                                          ages_softmaxs,
                                                                          xs_pl,
                                                                          diag_s_pl,
                                                                          ages_s_pl,
                                                                          training_time_pl=training_time_placeholder,
                                                                          directly_feed_clf_pl=directly_feed_clf_pl,
                                                                          images=images_val,
                                                                          labels_list=[labels_val, ages_val],
                                                                          clf_batch_size=2 * exp_config.batch_size,
                                                                          do_ordinal_reg=exp_config.age_ordinal_regression,
                                                                          selection_indices=source_images_val_ind)


                feed_dict_val = {
                    val_error_clf_: val_loss,
                    val_diag_f1_score_: val_diag_f1,
                    val_ages_f1_score_: val_ages_f1,
                    val_disc_loss_pl: d_loss_val_avg,
                    val_gen_loss_pl: g_loss_val_avg
                }

                validation_summary_msg = sess.run(val_summary, feed_dict=feed_dict_val)
                summary_writer.add_summary(validation_summary_msg, step)
                summary_writer.flush()

                # save best variables (if discriminator loss is the lowest yet)
                if d_loss_val_avg <= best_d_loss:
                    best_d_loss = d_loss_val_avg
                    best_file = os.path.join(log_dir, 'model_best_d_loss.ckpt')
                    saver_best_disc.save(sess, best_file, global_step=step)
                    logging.info('Found new best discriminator loss on validation set! - %f -  Saving model_best_d_loss.ckpt' % best_d_loss)

                if val_diag_f1 >= best_diag_f1_score:
                    best_diag_f1_score = val_diag_f1
                    best_file = os.path.join(log_dir, 'model_best_diag_f1.ckpt')
                    saver_best_diag_f1.save(sess, best_file, global_step=step)
                    logging.info(
                        'Found new best DIAGNOSIS F1 score on validation set! - %f -  Saving model_best_diag_f1.ckpt' % val_diag_f1)

                if val_ages_f1 >= best_ages_f1_score:
                    best_ages_f1_score = val_ages_f1
                    best_file = os.path.join(log_dir, 'model_best_ages_f1.ckpt')
                    saver_best_ages_f1.save(sess, best_file, global_step=step)
                    logging.info(
                        'Found new best AGES F1 score on validation set! - %f -  Saving model_best_ages_f1.ckpt' % val_ages_f1)

                if val_loss <= best_val:
                    best_val = val_loss
                    best_file = os.path.join(log_dir, 'model_best_xent.ckpt')
                    saver_best_xent.save(sess, best_file, global_step=step)
                    logging.info(
                        'Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

                logging.info("[Validation], generator loss: %g, discriminator_loss: %g" % (g_loss_val_avg, d_loss_val_avg))

            # Write the summaries and print an overview fairly often.
            if step % exp_config.save_frequency == 0:

                saver_latest.save(sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)

        sess.close()


def image_tensor_shape(batch_size):
    return [batch_size] + list(exp_config.image_size) + [exp_config.n_channels]


def image_placeholder(batch_size, name):
    return tf.placeholder(tf.float32, image_tensor_shape(batch_size), name=name)


def placeholders_clf(batch_size, scope_name):
    with tf.variable_scope(scope_name):
        labels_tensor_shape = [batch_size]

        if exp_config.age_ordinal_regression:
            ages_tensor_shape = [batch_size, len(exp_config.age_bins)]
        else:
            ages_tensor_shape = [batch_size]

        images_pl = image_placeholder(batch_size, 'images')
        diag_pl = tf.placeholder(tf.uint8, shape=labels_tensor_shape, name='labels')
        ages_pl = tf.placeholder(tf.uint8, shape=ages_tensor_shape, name='ages')
        return images_pl, diag_pl, ages_pl


def concatenate_clf_input(images_list, diag_list, ages_list, scope_name='fs_concat'):
    with tf.variable_scope(scope_name):
        images = tf.concat(images_list, axis=0, name='images')
        diag = tf.concat(diag_list, axis=0, name='diagnose')
        ages = tf.concat(ages_list, axis=0, name='ages')
        return images, diag, ages



def do_eval_gan(sess, losses, images_s_pl, images_t_pl, training_time_placeholder, images, source_images_ind,
                target_images_ind, batch_size=exp_config.batch_size, num_batches=exp_config.num_val_batches):
    '''
    Function for running the evaluations of the gan every X iterations on the training and validation sets.
    :param sess: The current tf session
    :param losses: list of loss placeholders
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode.
    :param images: A numpy array or h5py dataset containing the images
    :param batch_size: The batch_size to use.
    :return: The average loss (as defined in the experiment), and the average dice over all `images`.
    '''

    s_sampler_val = iterate_minibatches_endlessly(images,
                                                  batch_size=batch_size,
                                                  exp_config=exp_config,
                                                  selection_indices=source_images_ind)
    t_sampler_val = iterate_minibatches_endlessly(images,
                                                  batch_size=batch_size,
                                                  exp_config=exp_config,
                                                  selection_indices=target_images_ind)

    # evaluate the validation batch with batch_size images (from each domain) at a time
    loss_val_array = np.empty((num_batches, len(losses)), dtype=np.float32)
    for batch_ind in range(exp_config.num_val_batches):
        x_t = next(t_sampler_val)
        x_s = next(s_sampler_val)
        loss_val = sess.run(
            losses, feed_dict={images_s_pl: x_s,
                               images_t_pl: x_t,
                               training_time_placeholder: False})
        loss_val_array[batch_ind, :] = np.array(loss_val)

    loss_val_avg = np.mean(loss_val_array, axis=0)
    logging.info(losses)
    logging.info(num_batches)
    logging.info('average val loss: ' + str(loss_val_avg.tolist()))

    return loss_val_avg.tolist()


def do_eval_classifier(sess, eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs, images_s_pl, diag_labels_pl,
                       ages_pl, training_time_pl, directly_feed_clf_pl, images, labels_list, clf_batch_size, do_ordinal_reg,
                       selection_indices=None):

    '''
    Function for running the evaluations every X iterations on the training and validation sets.
    :param sess: The current tf session
    :param eval_loss: The placeholder containing the eval loss
    :param images_pl: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_pl: Placeholder toggling the training/testing mode.
    :param images: A numpy array or h5py dataset containing the images
    :param labels_list: A numpy array or h45py dataset containing the corresponding labels
    :param clf_batch_size: The batch_size to use.
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
                                     batch_size=clf_batch_size,
                                     selection_indices=selection_indices,
                                     augmentation_function=None,
                                     exp_config=exp_config):  # No aug in evaluation
        # As before you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
        # but at the risk of not catching exceptions

        x, [y, a] = batch

        if y.shape[0] < clf_batch_size:
            continue

        feed_dict = {images_s_pl: x,
                     diag_labels_pl: y,
                     ages_pl: a,
                     training_time_pl: False,
                     directly_feed_clf_pl: False}

        c_d_loss, c_a_loss, c_d_preds, c_a_softmaxs = sess.run([eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs], feed_dict=feed_dict)

        # This converts the labels back into the original format. I.e. [0,1,1,0] will become [0,2,2,0] again if
        # 1 didn't exist in the dataset.
        c_d_preds = [exp_config.label_list[pp] for pp in c_d_preds]
        y_gts = [exp_config.label_list[pp] for pp in y]

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

    # check whether the labels are in {0, 2} as expected
    logging.info('diagnose predictions and ground truth:')
    logging.info(predictions_diag)
    assert all([label in {0, 2} for label in predictions_diag])
    logging.info(predictions_diag_gt)
    assert all([label in {0, 2} for label in predictions_diag_gt])

    f1_diag_score = f1_score(np.asarray(predictions_diag_gt), np.asarray(predictions_diag), pos_label=2, average='binary')  # micro is overall, macro doesn't take class imbalance into account
    # f1_ages_score = f1_score(np.asarray(predictions_ages_gt), np.asarray(predictions_ages), average='micro')  # micro is overall, macro doesn't take class imbalance into account

    f1_ages_score = np.mean(np.abs(np.asarray(predictions_ages, dtype=np.int32) - np.asarray(predictions_ages_gt,  dtype=np.int32)))

    logging.info('  Average loss: %0.04f, diag f1_score: %0.04f, age f1_score %0.04f' % (avg_loss, f1_diag_score, f1_ages_score))

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


