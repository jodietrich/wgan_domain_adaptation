# Simple loop for displaying predictions for random slices from the test dataset
#
# Usage:
#
# python test_loop.py path/to/experiment_logs
#
#
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
import utils
import image_utils

from scipy import misc


#######################################################################

# from experiments import dcgan_improved_train as exp_config
# from experiments import dcgan_fcn_improved_train as exp_config

# LEARN SCALE
from experiments import dcgan_fcn_bn_improved_train as exp_config
experiment_name = 'dcgan_fcn_bn_improved_train_fcn_scale_wtest'
from data import mnist_scale as data
invert_img = False

# LEARN INVERSION
# from experiments import dcgan_fcn_bn_improved_train as exp_config
# experiment_name = 'dcgan_fcn_bn_improved_train_fcn_invert_wtest'
# from data import mnist_invert as data
# invert_img = True

#######################################################################

log_dir = os.path.join(sys_config.log_root, experiment_name)


def load_image(path='./images/im_w.png', invert=False):

    img = misc.imread(path)
    img = np.mean(img, axis=2)
    img = np.float32(img) / np.max(img)

    if invert:
        img = -1.0*img + 1  # invert
        noise = np.random.uniform(0.0, 0.2, img.shape)
        img -= noise
        img[img < 0.0] = 0.0

    return np.reshape(img, [1, 28, 28, 1])


def main():

    # Load data
    nets = exp_config.model_handle

    im_s = exp_config.data_shape

    training_placeholder = tf.placeholder(tf.bool, name='training_phase')
    x_pl = tf.placeholder(tf.float32, [1, im_s[0], im_s[1], 1], name='x')
    z_pl = tf.placeholder(tf.float32, [1, im_s[0], im_s[1], 1], name='z')

    x_pl_ = nets.generator(z_pl, training_placeholder)

    d_pl = nets.discriminator(x_pl, training_placeholder, scope_reuse=False)
    d_pl_ = nets.discriminator(x_pl_, training_placeholder, scope_reuse=True)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    z_sampler = data.TargetDataSampler()


    with tf.Session() as sess:

        sess.run(init)

        checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        saver.restore(sess, checkpoint_path)

        while True:

            # z_in = z_sampler.get_validation_batch(2)[np.newaxis,0,...]
            z_in = load_image('./images/im_a.png', invert=invert_img)

            x_ = sess.run(x_pl_, feed_dict={z_pl: z_in, training_placeholder: False})

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(np.squeeze(z_in), cmap='gray')
            ax2 = fig.add_subplot(122)
            ax2.imshow(np.squeeze(x_), cmap='gray')

            plt.show()



if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    # parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    # args = parser.parse_args()
    #
    # base_path = sys_config.project_root
    #
    # model_path = os.path.join(base_path, args.EXP_PATH)
    # config_file = glob.glob(model_path + '/*py')[0]
    # config_module = config_file.split('/')[-1].rstrip('.py')
    #
    # exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # init_iteration = main(exp_config=exp_config)
    init_iteration = main()
