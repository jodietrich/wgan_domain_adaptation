import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

experiment_name = 'residual_identity_gen_bs2_bn_std_disc'

# paths
data_root = sys_config.data_root
preproc_folder = os.path.join(sys_config.project_root,'data/adni/preprocessed')

# Model settings
model_handle = model_zoo.ResNet_gen_bs2_bn

# Data settings
image_size = (128, 160, 112)
target_resolution =  (1.5, 1.5, 1.5)
label_list = (0,2)  # 0 - normal, 1 - mci, 2 - alzheimer's
source_field_strength = 3.0 # magnetic field strength in T of pictures in the source-domain
target_field_strength = 1.5 # magnetic field strength in T of pictures in the target-domain
n_channels = 1

# visualization settings
image_z_slice = 50

# Training settings
batch_size = 2
val_batch_size = batch_size # must be smaller than the smallest validation set
learning_rate = 1e-4
optimizer_handle = tf.train.AdamOptimizer

# Improved training settings
improved_training = True
scale=10.0

# Regularisation settings
w_reg_gen_l1 = 0.0
w_reg_disc_l1 = 0.0
w_reg_gen_l2 = 0.0
w_reg_disc_l2 = 0.0

# Rarely changed settings
max_iterations = 100000
save_frequency = 200
validation_frequency = 100
update_tensorboard_frequency = 10
