

__author__ = 'jdietric'

import itertools
import logging
import numpy as np
import os
import tensorflow as tf

import config.system as sys_config
import utils
import adni_data_loader_all
import experiments.gan.standard_parameters as std_params


def build_gen_graph(img_tensor_shape, gan_config):
    # noise_shape
    generator = gan_config.generator
    graph_generator = tf.Graph()
    with graph_generator.as_default():
        # source image (batch size = 1)
        xs_pl = tf.placeholder(tf.float32, img_tensor_shape, name='xs_pl')

        if gan_config.use_generator_input_noise:
            noise_pl = tf.placeholder(tf.float32, (1, gan_config.generator_input_noise_shape[1]), name='z_noise')
        else:
            noise_pl = None

        # generated fake image batch
        xf = generator(xs=xs_pl, z_noise=noise_pl, training=False)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a savers for writing training checkpoints.
        saver = tf.train.Saver()
        return graph_generator, xs_pl, noise_pl, xf, init, saver


def generate_with_noise(gan_experiment_path_list, noise_list,
                        image_saving_indices=set(), image_saving_path3d=None, image_saving_path2d=None):
    """

    :param gan_experiment_path_list: list of GAN experiment paths to be evaluated. They must all have the same image settings and source/target field strengths as the classifier
    :param clf_experiment_path: AD classifier used
    :param image_saving_indices: set of indices of the images to be saved
    :param image_saving_path: where to save the images. They are saved in subfolders for each experiment
    :return:
    """

    batch_size = 1
    logging.info('batch size %d is used for everything' % batch_size)

    for gan_experiment_path in gan_experiment_path_list:
        gan_config, logdir_gan = utils.load_log_exp_config(gan_experiment_path)

        gan_experiment_name = gan_config.experiment_name

        log_dir_ending = logdir_gan.split('_')[-1]
        continued_experiment = (log_dir_ending == 'cont')
        if continued_experiment:
            gan_experiment_name += '_cont'

        # make sure the noise has the right dimension
        assert gan_config.use_generator_input_noise
        assert gan_config.generator_input_noise_shape[1:] == std_params.generator_input_noise_shape[1:]

        # Load data
        data = adni_data_loader_all.load_and_maybe_process_data(
            input_folder=gan_config.data_root,
            preprocessing_folder=gan_config.preproc_folder,
            size=gan_config.image_size,
            target_resolution=gan_config.target_resolution,
            label_list=gan_config.label_list,
            offset=gan_config.offset,
            rescale_to_one=gan_config.rescale_to_one,
            force_overwrite=False
        )

        # extract images and indices of source/target images for the test set
        images_test = data['images_test']

        im_s = gan_config.image_size

        img_tensor_shape = [batch_size, im_s[0], im_s[1], im_s[2], 1]

        logging.info('\nGAN Experiment (%.1f T to %.1f T): %s' % (gan_config.source_field_strength,
                                                              gan_config.target_field_strength, gan_experiment_name))
        logging.info(gan_config)
        # open GAN save file from the selected experiment

        # prevents ResourceExhaustError when a lot of memory is used
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.

        source_indices = []
        target_indices = []
        for i, field_strength in enumerate(data['field_strength_test']):
            if field_strength == gan_config.source_field_strength:
                source_indices.append(i)
            elif field_strength == gan_config.target_field_strength:
                target_indices.append(i)

        num_source_images = len(source_indices)
        num_target_images = len(target_indices)

        logging.info('Data summary:')
        logging.info(' - Images:')
        logging.info(images_test.shape)
        logging.info(images_test.dtype)
        logging.info(' - Domains:')
        logging.info('number of source images: ' + str(num_source_images))
        logging.info('number of target images: ' + str(num_target_images))

        # save real images
        source_image_path = os.path.join(image_saving_path3d, 'source')
        utils.makefolder(source_image_path)
        sorted_saving_indices = sorted(image_saving_indices)

        source_saving_indices = [source_indices[index] for index in sorted_saving_indices]
        for source_index in source_saving_indices:
            source_img_name = 'source_img_%.1fT_%d.nii.gz' % (gan_config.source_field_strength, source_index)
            utils.create_and_save_nii(images_test[source_index], os.path.join(source_image_path, source_img_name))
            logging.info(source_img_name + ' saved')

        logging.info('source images saved')

        logging.info('loading GAN')
        # open the latest GAN savepoint
        init_checkpoint_path_gan, last_gan_step = utils.get_latest_checkpoint_and_step(logdir_gan, 'model.ckpt')

        logging.info(init_checkpoint_path_gan)

        # build a separate graph for the generator
        graph_generator, generator_img_pl, z_noise_pl, x_fake_op, init_gan_op, saver_gan = build_gen_graph(img_tensor_shape, gan_config)

        # Create a session for running Ops on the Graph.
        sess_gan = tf.Session(config=config, graph=graph_generator)

        # Run the Op to initialize the variables.
        sess_gan.run(init_gan_op)
        saver_gan.restore(sess_gan, init_checkpoint_path_gan)

        # path where the generated images are saved
        experiment_generate_path_3d = os.path.join(image_saving_path_3d, gan_experiment_name + ('_%.1fT_source' % gan_config.source_field_strength))
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path_3d)

        # path where the generated image 2d cuts are saved
        experiment_generate_path_2d = os.path.join(image_saving_path_2d, gan_experiment_name + (
        '_%.1fT_source' % gan_config.source_field_strength))
        # make a folder for the generated images
        utils.makefolder(experiment_generate_path_2d)

        logging.info('image generation begins')
        generated_pred = []
        batch_beginning_index = 0
        # loops through all images from the source domain
        for image_index, curr_img in zip(source_saving_indices, itertools.compress(images_test, source_saving_indices)):
            img_folder_name = 'image_test%d' % image_index
            curr_img_path_3d = os.path.join(experiment_generate_path_3d, img_folder_name)
            utils.makefolder(curr_img_path_3d)
            curr_img_path_2d = os.path.join(experiment_generate_path_2d, img_folder_name)
            utils.makefolder(curr_img_path_2d)
            # save source image
            source_img_name = 'source_img'
            utils.save_image_and_cut(np.squeeze(curr_img), source_img_name, curr_img_path_3d, curr_img_path_2d, vmin=-1, vmax=1)
            logging.info(source_img_name + ' saved')
            img_list = []
            for noise_index, noise in enumerate(noise_list):
                fake_img = sess_gan.run(x_fake_op, feed_dict={generator_img_pl: np.reshape(curr_img, img_tensor_shape),
                                                              z_noise_pl: noise})
                fake_img = np.squeeze(fake_img)
                # make sure the dimensions are right
                assert len(fake_img.shape) == 3

                img_list.append(fake_img)

                generated_img_name = 'generated_img_noise_%d' % (noise_index)
                utils.save_image_and_cut(np.squeeze(fake_img), generated_img_name, curr_img_path_3d, curr_img_path_2d, vmin=-1, vmax=1)
                logging.info(generated_img_name + ' saved')

                # save the difference g(xs)-xs
                difference_image_gs = np.squeeze(fake_img) - curr_img
                difference_img_name = 'difference_img_noise_%d' % (noise_index)
                utils.save_image_and_cut(difference_image_gs, difference_img_name, curr_img_path_3d, curr_img_path_2d, vmin=-1, vmax=1)
                logging.info(difference_img_name + ' saved')

            # works because axis 0
            all_imgs = np.stack(img_list, axis=0)
            std_img = np.std(all_imgs, axis=0)
            std_img_name = 'std_img'
            utils.save_image_and_cut(std_img, std_img_name, curr_img_path_3d, curr_img_path_2d, vmin=0, vmax=1)
            logging.info(std_img_name + ' saved')

        logging.info('generated all images for %s' % (gan_experiment_name))



def generate_noise_list(noise_shape, seed_list=range(10), noise_function=lambda shape: np.random.uniform(low=-1.0, high=1.0, size=shape)):
    # creates a list of random ndarrays with the given shape. The list has length len(seed_list) and uses one seed for each list element
    noise_list = []
    for seed in seed_list:
        np.random.seed(seed)
        noise = noise_function(noise_shape)
        noise_list.append(noise)
        logging.info('seed: ' + str(seed))
        logging.info('noise: ' + str(noise))
    return noise_list

def generate_points_on_line(total_points):
    noise_dimension = 10
    start = np.array([-1] * noise_dimension)
    end = np.array([1] * noise_dimension)
    place_parameter_list = np.linspace(0, 1, num=total_points, endpoint=True)
    points = [np.expand_dims(start*(1-r) + end*r, axis=0) for r in place_parameter_list]
    logging.info(points)
    return points


if __name__ == '__main__':
    # settings
    # experiment lists to choose from
    gan_experiment_list1 = [
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s3_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s3_final_i1',
        'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s15_final_i1',
        'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s15_final_i1'
    ]

    joint_experiment_list1 = [
        'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1_cont',
        'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s15_bs6_i1_cont',
        'joint_genval_gan_residual_gen_n8b4_disc_n8_dropout_keep0.9_10_noise_1e4l1_clfWeight1e5_all_small_final_s3_bs6_i1_cont'
    ]

    experiment_list = joint_experiment_list1
    joint = True  # joint or separate training
    if joint:
        gan_log_root = os.path.join(sys_config.log_root, 'joint/final')
    else:
        gan_log_root = os.path.join(sys_config.log_root, 'gan/final')
    # image_saving_path = os.path.join(sys_config.project_root,'data/generated_images/final/const_noise')
    image_saving_path_3d = os.path.join(sys_config.project_root, 'data/generated_images/final/interpolated')
    image_saving_path_2d = os.path.join(sys_config.project_root, 'data/generated_images/final/interpolated/coronal_2d')
    image_saving_indices = set(range(0, 220, 20))
    seed_list = range(10)

    # put paths for experiments together
    log_path_list = [os.path.join(gan_log_root, gan_name) for gan_name in experiment_list]

    # noise_list = generate_noise_list(noise_shape=(1, std_params.generator_input_noise_shape[1]), seed_list=seed_list)
    noise_list = generate_points_on_line(10)

    generate_with_noise(gan_experiment_path_list=log_path_list,
                        noise_list=noise_list,
                        image_saving_indices=image_saving_indices,
                        image_saving_path3d=image_saving_path_3d,
                        image_saving_path2d=image_saving_path_2d)
