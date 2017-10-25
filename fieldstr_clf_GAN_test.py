__author__ = 'jdietric'

import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import shutil
import random

import config.system as sys_config

import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import data_utils


#######################################################################

from experiments import residual_gen_bs2 as exp_config

#######################################################################


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
