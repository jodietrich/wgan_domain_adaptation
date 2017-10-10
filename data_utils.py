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
    def __init__(self, train_images, validation_images):
        self.shape = list(exp_config.image_size) + [exp_config.n_channels] # [x, y, z, #channels]
        self.train_data = train_images
        self.validation_data = validation_images
        self.current_index = 0 # Index where the next batch starts

# batch is sequential images. Ask Christian if reshuffling is needed.
    def __call__(self, batch_size):
        next_index = (self.current_index + batch_size) % len(self.train_data)
        batch = self.train_data[self.current_index:next_index]
        self.current_index = next_index
        return self.data2img(batch)

    def get_validation_batch(self, batch_size):
        next_index = (self.current_index + batch_size) % len(self.train_data)
        batch = self.validation_data[self.current_index:next_index]
        self.current_index = next_index
        return self.data2img(batch)

    def data2img(self, data):
        return np.reshape(data, [len(data)] + self.shape)
