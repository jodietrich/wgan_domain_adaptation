import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

from experiments.standard_parameters import *

experiment_name = 'std_cnn_identity_gen_batchsize1_v2'

# Model settings
model_handle = model_zoo.Std_CNN_bs1

# Training settings
batch_size = 1
val_batch_size = batch_size # must be smaller than the smallest validation set

