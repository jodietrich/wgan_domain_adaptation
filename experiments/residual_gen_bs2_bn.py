import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

from experiments.standard_parameters import *

experiment_name = 'residual_identity_gen_bs2_bn_std_disc_2_v4'

# Model settings
model_handle = model_zoo.Res_Gen_bs2_bn
