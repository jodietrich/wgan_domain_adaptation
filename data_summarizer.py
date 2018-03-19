# Authors:
# Jonathan Dietrich

# summarize MRI data set

import logging
import time

import numpy as np
import os.path
import shutil
import tensorflow as tf
from sklearn.metrics import f1_score

import adni_data_loader_all
import config.system as sys_config
import clf_model_multitask as model_mt
import utils
from batch_generator_list import iterate_minibatches
import data_utils
import gan_model

from experiments.adni_clf import allconv_bn as exp_config

# Load data
data = adni_data_loader_all.load_and_maybe_process_data(
    input_folder=exp_config.data_root,
    preprocessing_folder=exp_config.preproc_folder,
    size=exp_config.image_size,
    target_resolution=exp_config.target_resolution,
    label_list=exp_config.label_list,
    offset=exp_config.offset,
    rescale_to_one=exp_config.rescale_to_one,
    force_overwrite=False
)

# the following are HDF5 datasets, not numpy arrays
labels_train = data['diagnosis_train']
ages_train = data['age_train']

images_train, source_images_train_ind, target_images_train_ind, \
images_val, source_images_val_ind, target_images_val_ind = data_utils.get_images_and_fieldstrength_indices(data, exp_config.source_field_strength, exp_config.target_field_strength)

data_summary_dict = data_utils.data_summary(data)
print(data_summary_dict)
