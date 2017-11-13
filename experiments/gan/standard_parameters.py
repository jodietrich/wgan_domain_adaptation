# these parameters are used for every experiment unless otherwise specified in the experiment file

import model_zoo
import tensorflow as tf
import config.system as sys_config
import os


# paths
data_root = sys_config.data_root
preproc_folder = os.path.join(sys_config.project_root,'data/adni/preprocessed')
log_folder = 'gan/all_small_images'

# model settings
gen_hidden_layers = 2
gen_filters = 16

# Data settings
# image_size = (128, 160, 112)
# target_resolution =  (1.5, 1.5, 1.5)
image_size = (64, 80, 64)
target_resolution =  (1.5, 1.5, 1.5)
offset = (0, 0, -10)
label_list = (0, 2)  # 0 - normal, 1 - mci, 2 - alzheimer's
nlabels = len(label_list)
source_field_strength = 3.0 # magnetic field strength in T of pictures in the source-domain
target_field_strength = 1.5 # magnetic field strength in T of pictures in the target-domain
n_channels = 1
image_range = (-1, 1)
# image_range = (-0.512, 2.985) # approximately 1st percentile to 99th percentile of preprocessed ADNI images

# visualization settings
cut_axis = 2  # axis perpendicular to the cut plane (x=0, y=1, z=2)
cut_index = 50  # index of the cut for visualization
diff_threshold = 1  # maximum absolute value for visualization of difference between images. Gets mapped to 255

# Training settings
batch_size = 20
num_val_batches = 5 # of batches used for validation. Validation happens with a set of size batch_size*num_val_batches
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
w_reg_img_dist_l1 = 0.0  # weight of l1 distance to source image in gen loss

# Rarely changed settings
max_iterations = 100000
save_frequency = 200
validation_frequency = 100
update_tensorboard_frequency = 10
max_epochs = 20000
