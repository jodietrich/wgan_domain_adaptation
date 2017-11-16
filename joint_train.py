# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import shutil

import config.system as sys_config
import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader_all
import data_utils
from batch_generator_list import iterate_minibatches_endlessly
import model_multitask as model_mt



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
        rescale_to_one=True,
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
    classifier = exp_config.model_handle

    s_sampler_train = iterate_minibatches_endlessly(images_train,
                                                    batch_size=exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    labels_list=[labels_train, ages_train],
                                                    selection_indices=source_images_train_ind)
    t_sampler_train = iterate_minibatches_endlessly(images_train,
                                                    batch_size=exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    labels_list=[labels_train, ages_train],
                                                    selection_indices=target_images_train_ind)


    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        image_tensor_shape_gan = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.n_channels]

        training_time_placeholder = tf.placeholder(tf.bool, shape=[], name='training_time')

        # GAN

        # target image batch
        xt_pl = tf.placeholder(tf.float32, image_tensor_shape_gan, name='x')

        # source image batch
        xs_pl = tf.placeholder(tf.float32, image_tensor_shape_gan, name='z')

        # generated fake image batch
        xf_pl = generator(xs_pl, training_time_placeholder)

        # difference between generated and source images
        diff_img_pl = xf_pl - xs_pl

        # visualize the images by showing one slice of them in the z direction
        tf.summary.image('sample_outputs', tf_utils.put_kernels_on_grid3d(xf_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_xt', tf_utils.put_kernels_on_grid3d(xt_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_xs', tf_utils.put_kernels_on_grid3d(xs_pl, exp_config.cut_axis,
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

        learning_rate_placeholder = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        if exp_config.momentum is not None:
            optimiser = exp_config.optimizer_handle(learning_rate=learning_rate_placeholder,
                                                    momentum=exp_config.momentum)
        else:
            optimiser = exp_config.optimizer_handle(learning_rate=learning_rate_placeholder)

        # nr means no regularization, meaning the loss without the regularization term
        discriminator_train_op, generator_train_op, \
        disc_loss_pl, gen_loss_pl, \
        disc_loss_nr_pl, gen_loss_nr_pl = model.training_ops(d_pl, d_pl_,
                                                             optimizer_handle=exp_config.optimizer_handle,
                                                             learning_rate=exp_config.learning_rate,
                                                             l1_img_dist=dist_l1,
                                                             w_reg_img_dist_l1=exp_config.w_reg_img_dist_l1,
                                                             w_reg_gen_l1=exp_config.w_reg_gen_l1,
                                                             w_reg_disc_l1=exp_config.w_reg_disc_l1,
                                                             w_reg_gen_l2=exp_config.w_reg_gen_l2,
                                                             w_reg_disc_l2=exp_config.w_reg_disc_l2,
                                                             d_hat=d_hat, x_hat=x_hat, scale=exp_config.scale)


        # Build the operation for clipping the discriminator weights
        d_clip_op = model.clip_op()

        # Put L1 distance of generated image and original image on summary
        dist_l1_summary_op = tf.summary.scalar('L1_distance_to_source_img', dist_l1)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # validation summaries
        val_disc_loss_pl = tf.placeholder(tf.float32, shape=[], name='disc_val_loss')
        disc_val_summary_op = tf.summary.scalar('validation_discriminator_loss', val_disc_loss_pl)

        val_gen_loss_pl = tf.placeholder(tf.float32, shape=[], name='gen_val_loss')
        gen_val_summary_op = tf.summary.scalar('validation_generator_loss', val_gen_loss_pl)

        val_summary_op_gan = tf.summary.merge([disc_val_summary_op, gen_val_summary_op])

        # Classifier ----------------------------------------------------------------------------------------

        labels_tensor_shape = [exp_config.batch_size]

        if exp_config.age_ordinal_regression:
            ages_tensor_shape = [exp_config.batch_size, len(exp_config.age_bins)]
        else:
            ages_tensor_shape = [exp_config.batch_size]

        diag_s_pl = tf.placeholder(tf.uint8, shape=labels_tensor_shape, name='labels')
        ages_s_pl = tf.placeholder(tf.uint8, shape=ages_tensor_shape, name='ages')

        # combine source and generated images into one minibatch for the classifier
        x_clf_all = tf.concat([xf_pl, xs_pl], axis=0)
        diag_all = tf.concat([diag_s_pl, diag_s_pl], axis=0)
        ages_all = tf.concat([ages_s_pl, ages_s_pl], axis=0)

        tf.summary.scalar('learning_rate', learning_rate_placeholder)

        # Build a Graph that computes predictions from the inference model.
        diag_logits, ages_logits = exp_config.model_handle(x_clf_all,
                                                           nlabels=exp_config.nlabels,
                                                           training=training_time_placeholder,
                                                           n_age_thresholds=len(exp_config.age_bins),
                                                           bn_momentum=exp_config.bn_momentum)

        # Add to the Graph the Ops for loss calculation.

        [classifier_loss, diag_loss, age_loss, weights_norm] = model_mt.loss(diag_logits,
                                                                  ages_logits,
                                                                  diag_all,
                                                                  ages_all,
                                                                  nlabels=exp_config.nlabels,
                                                                  weight_decay=exp_config.weight_decay,
                                                                  diag_weight=exp_config.diag_weight,
                                                                  age_weight=exp_config.age_weight,
                                                                  use_ordinal_reg=exp_config.age_ordinal_regression,
                                                                  ordinal_reg_weights=ordinal_reg_weights)

        tf.summary.scalar('classifier loss', classifier_loss)
        tf.summary.scalar('diag_loss', diag_loss)
        tf.summary.scalar('age_loss', age_loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        # TODO: make the train operation in a separate function and make sure these are the right variables
        train_variables = tf.trainable_variables()
        classifier_variables = [v for v in train_variables if v.name.startswith("classifier")]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_clf_op = optimiser.minimize(classifier_loss, var_list=classifier_variables)

        eval_diag_loss, eval_ages_loss, pred_labels, ages_softmaxs = model_mt.evaluation(diag_logits, ages_logits,
                                                                                         diag_all,
                                                                                         ages_all,
                                                                                         x_clf_all,
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

        # Classifier summary
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

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init)

        if continue_run:
            # Restore session
            saver_latest.restore(sess, init_checkpoint_path)

        curr_lr = exp_config.learning_rate

        # initialize value of lowest (i. e. best) discriminator loss
        best_d_loss = np.inf

        for step in range(init_step, exp_config.max_steps):

            start_time = time.time()

            # discriminator and classifier (task) training iterations
            d_iters = 5
            t_iters = 1
            if step % 500 == 0 or step < 25:
                d_iters = 100
            assert d_iters >= t_iters
            for iteration in range(max(d_iters, t_iters)):

                x_t, [diag_t, age_t] = next(t_sampler_train)
                x_s, [diag_s, age_s] = next(s_sampler_train)

                feed_dict_dc = {xs_pl: x_s,
                             xt_pl: x_t,
                             learning_rate_placeholder: curr_lr,
                             diag_s_pl: diag_s,
                             ages_s_pl: age_s,
                             training_time_placeholder: True}
                train_ops = []
                if iteration < t_iters:
                    # train classifier
                    train_ops.append(train_clf_op)
                if iteration < d_iters:
                    # train discriminator
                    train_ops.append(discriminator_train_op)
                sess.run(train_ops, feed_dict = feed_dict_dc)

                if not exp_config.improved_training:
                    sess.run(d_clip_op)

            elapsed_time = time.time() - start_time

            # train generator
            x_t = next(t_sampler_train)  # why not sample a new x??
            x_s = next(s_sampler_train)
            sess.run(generator_train_op,
                     feed_dict={xs_pl: x_s, xt_pl: x_t, training_time_placeholder: True})

            if step % exp_config.update_tensorboard_frequency == 0:

                x_t = next(t_sampler_train)
                x_s = next(s_sampler_train)

                g_loss_train, d_loss_train, summary_str = sess.run(
                        [gen_loss_nr_pl, disc_loss_nr_pl, summary_op], feed_dict={xs_pl: x_s, xt_pl: x_t, training_time_placeholder: False})

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                logging.info("[Step: %d], generator loss: %g, discriminator_loss: %g" % (step, g_loss_train, d_loss_train))
                logging.info(" - elapsed time for one step: %f secs" % elapsed_time)


            if step % exp_config.validation_frequency == 0:

                z_sampler_val = iterate_minibatches_endlessly(images_val,
                                                    batch_size=exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    selection_indices=source_images_val_ind)
                x_sampler_val = iterate_minibatches_endlessly(images_val,
                                                    batch_size=exp_config.batch_size,
                                                    exp_config=exp_config,
                                                    selection_indices=target_images_val_ind)

                # evaluate the validation batch with batch_size images (from each domain) at a time
                g_loss_val_list = []
                d_loss_val_list = []
                for _ in range(exp_config.num_val_batches):
                    x_t = next(x_sampler_val)
                    x_s = next(z_sampler_val)
                    g_loss_val, d_loss_val = sess.run(
                        [gen_loss_nr_pl, disc_loss_nr_pl], feed_dict={xs_pl: x_s,
                                                                      xt_pl: x_t,
                                                                      training_time_placeholder: False})
                    g_loss_val_list.append(g_loss_val)
                    d_loss_val_list.append(d_loss_val)

                g_loss_val_avg = np.mean(g_loss_val_list)
                d_loss_val_avg = np.mean(d_loss_val_list)

                validation_summary_str = sess.run(val_summary_op_gan, feed_dict={val_disc_loss_pl: d_loss_val_avg,
                                                                                 val_gen_loss_pl: g_loss_val_avg}
                                             )
                summary_writer.add_summary(validation_summary_str, step)
                summary_writer.flush()

                # save best variables (if discriminator loss is the lowest yet)
                if d_loss_val_avg <= best_d_loss:
                    best_d_loss = d_loss_val_avg
                    best_file = os.path.join(log_dir, 'model_best_d_loss.ckpt')
                    saver_best_disc.save(sess, best_file, global_step=step)
                    logging.info('Found new best discriminator loss on validation set! - %f -  Saving model_best_d_loss.ckpt' % best_d_loss)

                logging.info("[Validation], generator loss: %g, discriminator_loss: %g" % (g_loss_val_avg, d_loss_val_avg))

            # Write the summaries and print an overview fairly often.
            if step % exp_config.save_frequency == 0:

                saver_latest.save(sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)




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


