from experiments.gan import residual_gen_bs2 as exp_config

__author__ = 'jdietric'

import numpy as np

import adni_data_loader
import tensorflow as tf


#######################################################################

#######################################################################

print(1/7)

with tf.Graph().as_default():
    ones = tf.ones([2,2,2,2])
    expanded1 = tf.expand_dims(ones, -1)
    expanded2 = tf.expand_dims(ones)
    sess1 = tf.Session()
    with sess1.as_default():
        exp1 = expanded1.eval()
        exp2 = expanded2.eval()
        s1 = exp1.shape
        s2 = exp2.shape


# import data
data = adni_data_loader.load_and_maybe_process_data(
    input_folder=exp_config.data_root,
    preprocessing_folder=exp_config.preproc_folder,
    size=exp_config.image_size,
    target_resolution=exp_config.target_resolution,
    label_list = exp_config.label_list,
    force_overwrite=False
)

images_train = data['images_train']
images_val = data['images_val']

# make a list of 3T and 1.5T training/test data indices in the training/test image table
source_images_train_ind = []
target_images_train_ind = []
source_images_val_ind = []
target_images_val_ind = []

for train_ind in range(0, len(images_train)):
    field_str = data['field_strength_train'][train_ind]
    if field_str == exp_config.source_field_strength:
        source_images_train_ind.append(train_ind)
    elif field_str == exp_config.target_field_strength:
        target_images_train_ind.append(train_ind)

for val_ind in range(0, len(images_val)):
    field_str = data['field_strength_val'][val_ind]
    if field_str == exp_config.source_field_strength:
        source_images_val_ind.append(val_ind)
    elif field_str == exp_config.target_field_strength:
        target_images_val_ind.append(val_ind)

all_images = np.concatenate((images_train, images_val), axis=0)

source_images_train = images_train[source_images_train_ind]
target_images_train = images_train[target_images_train_ind]