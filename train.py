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
import adni_data_loader
import data_utils


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

#######################################################################
from experiments.gan import residual_gen_bs2 as exp_config
#######################################################################

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)


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
    data = adni_data_loader.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        label_list = exp_config.label_list,
        force_overwrite=False
    )

    # extract images and indices of source/target images for the training and validation set
    images_train, source_images_train_ind, target_images_train_ind,\
    images_val, source_images_val_ind, target_images_val_ind = adni_data_loader.get_images_and_fieldstrength_indices(
        data, exp_config.source_field_strength, exp_config.target_field_strength)

    generator = exp_config.generator
    discriminator = exp_config.discriminator

    z_sampler = data_utils.DataSampler(images_train, source_images_train_ind, images_val, source_images_val_ind)
    x_sampler = data_utils.DataSampler(images_train, target_images_train_ind, images_val, target_images_val_ind)


    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        im_s = exp_config.image_size

        training_placeholder = tf.placeholder(tf.bool, name='training_phase')

        # target image batch
        x_pl = tf.placeholder(tf.float32, [exp_config.batch_size, im_s[0], im_s[1], im_s[2], exp_config.n_channels], name='x')

        # source image batch
        z_pl = tf.placeholder(tf.float32, [exp_config.batch_size, im_s[0], im_s[1], im_s[2], exp_config.n_channels], name='z')

        # generated fake image batch
        x_pl_ = generator(z_pl, training_placeholder)

        # visualize the images by showing one slice of them in the z direction
        tf.summary.image('sample_outputs', tf_utils.put_kernels_on_grid3d(x_pl_, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_xs', tf_utils.put_kernels_on_grid3d(x_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        tf.summary.image('sample_zs', tf_utils.put_kernels_on_grid3d(z_pl, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range))

        # output of the discriminator for real image
        d_pl = discriminator(x_pl, training_placeholder, scope_reuse=False)

        # output of the discriminator for fake image
        d_pl_ = discriminator(x_pl_, training_placeholder, scope_reuse=True)

        d_hat = None
        x_hat = None
        if exp_config.improved_training:

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * x_pl + (1 - epsilon) * x_pl_
            d_hat = discriminator(x_hat, training_placeholder, scope_reuse=True)

        # nr means no regularization, meaning the loss without the regularization term
        discriminator_train_op, generator_train_op, \
        disc_loss_pl, gen_loss_pl, \
        disc_loss_nr_pl, gen_loss_nr_pl = model.training_ops(d_pl, d_pl_,
                                                             optimizer_handle=exp_config.optimizer_handle,
                                                             learning_rate=exp_config.learning_rate,
                                                             w_reg_gen_l1=exp_config.w_reg_gen_l1,
                                                             w_reg_disc_l1=exp_config.w_reg_disc_l1,
                                                             w_reg_gen_l2=exp_config.w_reg_gen_l2,
                                                             w_reg_disc_l2=exp_config.w_reg_disc_l2,
                                                             d_hat=d_hat, x_hat=x_hat, scale=exp_config.scale)


        # Build the operation for clipping the discriminator weights
        d_clip_op = model.clip_op()

        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # validation summaries
        val_disc_loss_pl = tf.placeholder(tf.float32, shape=[], name='disc_val_loss')
        disc_val_summary_op = tf.summary.scalar('validation_discriminator_loss', val_disc_loss_pl)

        val_gen_loss_pl = tf.placeholder(tf.float32, shape=[], name='gen_val_loss')
        gen_val_summary_op = tf.summary.scalar('validation_generator_loss', val_gen_loss_pl)

        val_summary_op = tf.summary.merge([disc_val_summary_op, gen_val_summary_op])

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver_latest = tf.train.Saver(max_to_keep=3)
        saver_best_disc = tf.train.Saver(max_to_keep=3)  # disc loss is scaled negative EM distance

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


        # initialize value of lowest (i. e. best) discriminator loss
        best_d_loss = np.inf

        for step in range(init_step, 1000000):

            start_time = time.time()

            # discriminator training iterations
            d_iters = 5
            if step % 500 == 0 or step < 25:
                d_iters = 100

            for _ in range(d_iters):

                x = x_sampler(exp_config.batch_size)
                z = z_sampler(exp_config.batch_size)

                # train discriminator
                sess.run(discriminator_train_op,
                         feed_dict={z_pl: z, x_pl: x, training_placeholder: True})

                if not exp_config.improved_training:
                    sess.run(d_clip_op)

            elapsed_time = time.time() - start_time

            # train generator
            z = z_sampler(exp_config.batch_size)  # why not sample a new x??
            x = x_sampler(exp_config.batch_size)
            sess.run(generator_train_op,
                     feed_dict={z_pl: z, x_pl: x, training_placeholder: True})

            if step % exp_config.update_tensorboard_frequency == 0:

                x = x_sampler(exp_config.batch_size)
                z = z_sampler(exp_config.batch_size)

                g_loss_train, d_loss_train, summary_str = sess.run(
                        [gen_loss_nr_pl, disc_loss_nr_pl, summary_op], feed_dict={z_pl: z, x_pl: x, training_placeholder: False})

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                logging.info("[Step: %d], generator loss: %g, discriminator_loss: %g" % (step, g_loss_train, d_loss_train))
                logging.info(" - elapsed time for one step: %f secs" % elapsed_time)


            if step % exp_config.validation_frequency == 0:

                x = x_sampler.get_validation_batch(exp_config.batch_size*exp_config.num_val_batches)
                z = z_sampler.get_validation_batch(exp_config.batch_size*exp_config.num_val_batches)

                # evaluate the validation batch with batch_size images (from each domain) at a time
                g_loss_val_list = []
                d_loss_val_list = []
                b_size = exp_config.batch_size
                for batch_num in range(exp_config.num_val_batches):
                    current_indices = slice(batch_num*b_size, batch_num*b_size + b_size)
                    g_loss_val, d_loss_val = sess.run(
                        [gen_loss_nr_pl, disc_loss_nr_pl], feed_dict={z_pl: z[current_indices],
                                                                      x_pl: x[current_indices],
                                                                      training_placeholder: False})
                    g_loss_val_list.append(g_loss_val)
                    d_loss_val_list.append(d_loss_val)

                g_loss_val_avg = np.mean(g_loss_val_list)
                d_loss_val_avg = np.mean(d_loss_val_list)

                validation_summary_str = sess.run(val_summary_op, feed_dict={val_disc_loss_pl: d_loss_val_avg,
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


