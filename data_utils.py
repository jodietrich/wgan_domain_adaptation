__author__ = 'jdietric'

from train import exp_config
import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import random

import config.system as sys_config

import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader


class DataSampler(object):
    def __init__(self, train_images, images_train_indices, validation_images, images_val_indices):
        self.shape = list(exp_config.image_size) + [exp_config.n_channels] # [x, y, z, #channels]
        self.train_data = train_images
        self.train_subset_ind = images_train_indices # indices of the subset of the training data that gets sampled
        self.validation_data = validation_images
        self.val_subset_ind = images_val_indices # indices of the subset of the training data that gets sampled

# get batch of random images out of the images with index in train_subset_ind
    def __call__(self, batch_size):
        batch_indices = sorted(random.sample(self.train_subset_ind, batch_size))
        batch = self.train_data[batch_indices]
        return self.data2img(batch)

# get batch of random images out of the images with index in val_subset_ind
    def get_validation_batch(self, batch_size):
        batch_indices = sorted(random.sample(self.val_subset_ind, batch_size))
        batch = self.train_data[batch_indices]
        return self.data2img(batch)

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)
