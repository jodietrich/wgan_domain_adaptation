__author__ = 'jdietric'

from train import exp_config
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
import data_sampler
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist_invert/raw')


class MriDataSampler(object):
    def __init__(self, train_images, validation_images):
        self.shape = list(exp_config.image_size).append(1) # [x, y, z, #channels]
        self.train_data = train_images
        self.validation_data = validation_images

    def __call__(self, batch_size):

        return self.data2img(mnist.train.next_batch(batch_size)[0])

    def get_validation_batch(self, batch_size):
        img = self.data2img(mnist.validation.next_batch(batch_size)[0])
        return img

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)