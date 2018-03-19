# Authors:
# Jonathan Dietrich

# Write details (index, RID, AD label) about the test set into a csv file 

import logging
import time

import numpy as np
import os.path
import shutil

import adni_data_loader_all
import config.system as sys_config
import clf_model_multitask as model_mt
import utils
from batch_generator_list import iterate_minibatches
import data_utils
import gan_model
from collections import OrderedDict
import csv


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

for tt in ['train','test','val']:
    print(len(np.unique(data['rid_%s' % tt])))

# make list of index, label, rid
test_labels = data['diagnosis_test']
logging.info(test_labels)
with open(os.path.join(sys_config.project_root, 'results/final/label_list.csv'), 'w+', newline='') as csvfile:
    fieldnames = ['rid', 'index', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind, label in enumerate(test_labels):
        writer.writerow({'index': ind, 'rid': data['rid_test'][ind], 'label': label})
