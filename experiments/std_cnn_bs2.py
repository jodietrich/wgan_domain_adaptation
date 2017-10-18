import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

from experiments.standard_parameters import *

experiment_name = 'std_cnn_identity_gen_no_batch_normalization'

# Model settings
model_handle = model_zoo.Std_CNN_bs2
