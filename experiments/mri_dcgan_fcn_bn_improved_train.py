import model_zoo
import tensorflow as tf
import config.system as sys_config
import os

experiment_name = 'dcgan_fcn_bn_improved_train_fcn_mri_1'

# paths
preproc_folder = os.path.join(sys_config.project_root,'data/adni')

# Model settings
model_handle = model_zoo.DCGAN_FCN_bn

# Data settings
image_size = (128, 160, 112)
target_resolution =  (1.5, 1.5, 1.5)
label_list = (0,2)  # 0 - normal, 1 - mci, 2 - alzheimer's

# Training settings
batch_size = 64
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
