__author__ = 'jdietric'

from train_gan import exp_config
import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import random

import config.system as sys_config

import gan_model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader

def get_images_and_fieldstrength_indices(data, source_field_strength, target_field_strength):
    """
    extract images and indices of source/target images for the training and validation set
    gives back indices instead of subsets to use hdf5 datasets instead of ndarrays (to save memory)
    :param data: hdf5 dataset of ADNI data
    :param source_field_strength: value of the magnetic field strength [T] in the source domain
    :param target_field_strength: value of the magnetic field strength [T] in the target domain
    :return:images_train: training images hdf5 dataset contains ndarray with shape [number_of_images, x, y, z]
            source_images_train_ind: indices of the images from the source domain in images_train
            target_images_train_ind: indices of the images from the target domain in images_train
            analogous for validation set
    """
    images_train = data['images_train']
    images_val = data['images_val']

    source_images_train_ind = []
    target_images_train_ind = []
    source_images_val_ind = []
    target_images_val_ind = []

    for train_ind, _ in enumerate(images_train):
        field_str = data['field_strength_train'][train_ind]
        if field_str == source_field_strength:
            source_images_train_ind.append(train_ind)
        elif field_str == target_field_strength:
            target_images_train_ind.append(train_ind)

    for val_ind, _ in enumerate(images_val):
        field_str = data['field_strength_val'][val_ind]
        if field_str == source_field_strength:
            source_images_val_ind.append(val_ind)
        elif field_str == target_field_strength:
            target_images_val_ind.append(val_ind)

    return images_train, source_images_train_ind, target_images_train_ind,\
           images_val, source_images_val_ind, target_images_val_ind


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
