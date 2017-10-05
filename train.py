# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import logging
import time

import numpy as np
import os.path
import tensorflow as tf

import config.system as sys_config

import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

#######################################################################

# from experiments import dcgan_improved_train as exp_config
# from experiments import dcgan_fcn_improved_train as exp_config
from experiments import mri_dcgan_fcn_bn_improved_train as exp_config

#######################################################################

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# import data
data = adni_data_loader.load_and_maybe_process_data(
    input_folder=exp_config.data_root,
    preprocessing_folder=exp_config.preproc_folder,
    size=exp_config.image_size,
    target_resolution=exp_config.target_resolution,
    label_list = exp_config.label_list,
    force_overwrite=False
)

# separate 3T and 1.5T images
images_train = data['images_train']
indices_3 = [f==3 for f in list(data['field_strength_train'])]
images_train_3T = images_train[0]

# not yet needed
# labels_train = data['diagnosis_train']

# separate 1.5T and 3T data
print(images_train_3T)
# print(list(images_train['field_strength_train']))

def run_training():

    logging.info('===== RUNNING EXPERIMENT ========')
    logging.info(exp_config.experiment_name)
    logging.info('=================================')

    nets = exp_config.model_handle

    x_sampler = data.SourceDataSampler()
    z_sampler = data.TargetDataSampler()


    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        im_s = exp_config.data_shape

        training_placeholder = tf.placeholder(tf.bool, name='training_phase')
        x_pl = tf.placeholder(tf.float32, [exp_config.batch_size, im_s[0], im_s[1], 1], name='x')
        z_pl = tf.placeholder(tf.float32, [exp_config.batch_size, im_s[0], im_s[1], 1], name='z')

        x_pl_ = nets.generator(z_pl, training_placeholder)

        tf.summary.image('sample_outputs', tf_utils.put_kernels_on_grid(x_pl_)
        )

        tf.summary.image('sample_xs', tf_utils.put_kernels_on_grid(x_pl)
        )

        tf.summary.image('sample_zs', tf_utils.put_kernels_on_grid(z_pl)
        )


        d_pl = nets.discriminator(x_pl, training_placeholder, scope_reuse=False)
        d_pl_ = nets.discriminator(x_pl_, training_placeholder ,scope_reuse=True)

        d_hat = None
        x_hat = None
        if exp_config.improved_training:

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * x_pl + (1 - epsilon) * x_pl_
            d_hat = nets.discriminator(x_hat, training_placeholder, scope_reuse=True)

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

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init)

        for step in range(1000000):

            start_time = time.time()

            d_iters = 5
            if step % 500 == 0 or step < 25:
                d_iters = 100

            for _ in range(d_iters):

                x = x_sampler(exp_config.batch_size)
                z = z_sampler(exp_config.batch_size)

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(np.squeeze(x[0,...]), cmap='gray')
                # plt.figure()
                # plt.imshow(np.squeeze(z[0,...]), cmap='gray')
                # plt.show()

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

                x = x_sampler.get_validation_batch(64)
                z = z_sampler.get_validation_batch(64)

                g_loss_val, d_loss_val = sess.run(
                    [gen_loss_nr_pl, disc_loss_nr_pl], feed_dict={z_pl: z, x_pl: x, training_placeholder: False})

                validation_summary_str = sess.run(val_summary_op, feed_dict={val_disc_loss_pl: d_loss_val,
                                                                             val_gen_loss_pl: g_loss_val}
                                             )
                summary_writer.add_summary(validation_summary_str, step)
                summary_writer.flush()

                logging.info("[Validation], generator loss: %g, discriminator_loss: %g" % (g_loss_val, d_loss_val))

            # Write the summaries and print an overview fairly often.
            if step % exp_config.save_frequency == 0:

                saver.save(sess, os.path.join(log_dir, 'model.ckpt'), global_step=step)




def main():

    run_training()


if __name__ == '__main__':

    main()


