import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

from experiments.standard_parameters import *

experiment_name = 'std_cnn_identity_gen_bs1_i1'

# Model settings
residual = False
batch_normalization = False
gen_hidden_layers = 3
gen_filters = 32

# Training settings
batch_size = 1
num_val_batches = 20 # must be smaller than the smallest validation set

# model to use
def generator(z, training, scope_name='generator'):
    return model_zoo.only_conv_generator(z, training, residual=residual, batch_normalization=batch_normalization,
                                         scope_name=scope_name, hidden_layers=gen_hidden_layers, filters=gen_filters)

discriminator = model_zoo.pool_fc_discriminator_bs1