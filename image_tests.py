__author__ = 'jdietric'

import logging
import time

import numpy as np
import os.path
import tensorflow as tf
import matplotlib.pyplot as plt

import config.system as sys_config

import model
from tfwrapper import utils as tf_utils
import utils
import adni_data_loader
import data_utils

import random


from experiments import residual_gen_bs2_bn as exp_config

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
all_images = np.concatenate((images_train, images_val), axis=0)

# # make a list of 3T and 1.5T training/test data indices in the training/test image table
# source_images_train_ind = []
# target_images_train_ind = []
# source_images_val_ind = []
# target_images_val_ind = []
#
# for train_ind in range(0, len(images_train)):
#     field_str = data['field_strength_train'][train_ind]
#     if field_str == exp_config.source_field_strength:
#         source_images_train_ind.append(train_ind)
#     elif field_str == exp_config.target_field_strength:
#         target_images_train_ind.append(train_ind)
#
# for val_ind in range(0, len(images_val)):
#     field_str = data['field_strength_val'][val_ind]
#     if field_str == exp_config.source_field_strength:
#         source_images_val_ind.append(val_ind)
#     elif field_str == exp_config.target_field_strength:
#         target_images_val_ind.append(val_ind)
#
# print(len(source_images_train_ind))
# print(len(target_images_train_ind))
# print(len(source_images_val_ind))
# print(len(source_images_val_ind))
#
min_element = np.amin(all_images)
max_element = np.amax(all_images)
mean = np.mean(all_images)
std = np.std(all_images)
median = np.median(all_images)
percentile5 = np.percentile(all_images, 5)
percentile95 = np.percentile(all_images, 95)
percentile2 = np.percentile(all_images, 2)
percentile98 = np.percentile(all_images, 98)
percentile1 = np.percentile(all_images, 1)
percentile99 = np.percentile(all_images, 99)

print('min: ' + str(min_element))
print('max: ' + str(max_element))
print('mean: ' + str(mean))
print('standard deviation: ' + str(std))
print('median: ' + str(std))
print('5th percentile: ' + str(percentile5))
print('95th percentile: ' + str(percentile95))
print('2nd percentile: ' + str(percentile2))
print('98th percentile: ' + str(percentile98))
print('1st percentile: ' + str(percentile1))
print('99th percentile: ' + str(percentile99))

images = np.expand_dims(all_images[3::50], 4)

# print('3D IMAGE')
# print(images)
#
# print('CUT BEFORE RESCALING')
# print(images[:, :, :, exp_config.cut_index, :])

sess = tf.Session()
with sess.as_default():
    images_tensor = tf.convert_to_tensor(images)

    print('AUTOMATIC RESCALING')
    autoimage = tf_utils.put_kernels_on_grid3d(images_tensor, 2, 50).eval()
    # print(np.squeeze(autoimage))
    plt.figure()
    plt.imshow(np.squeeze(autoimage), cmap='gray')

    print('MANUAL RESCALING')
    manimage1 = tf_utils.put_kernels_on_grid3d(images_tensor, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=exp_config.image_range).eval()
    manimage2 = tf_utils.put_kernels_on_grid3d(images_tensor, exp_config.cut_axis,
                                                                          exp_config.cut_index, rescale_mode='manual',
                                                                          input_range=(-0.512, 2.985)).eval()
    # print(np.squeeze(manimage))
    plt.figure()
    plt.imshow(np.squeeze(manimage1), cmap='gray')
    plt.figure()
    plt.imshow(np.squeeze(manimage2), cmap='gray')
    plt.show()