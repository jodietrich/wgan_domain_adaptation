import model_zoo
import tensorflow as tf
import config.system as sys_config
import os
import batch_augmentors

experiment_name = 'fclf_jiaxi_net_small_data'

# paths
log_folder = 'fclf'

# Model settings
model_handle = model_zoo.jia_xi_net_multitask_ordinal_bn
multi_task_model = True

# Data settings
# image_size = (128, 160, 112)
# target_resolution =  (1.5, 1.5, 1.5)
image_size = (64, 80, 64)
target_resolution =  (1.5, 1.5, 1.5)
offset = (0, 0, -10)
diagnosis_list = (0, 1, 2)  # 0 - normal, 1 - mci, 2 - alzheimer's
field_strength_list = (1.5, 3.0)
fs_label_list = (0, 1)
age_bins = (65, 70, 75, 80, 85)
nlabels = len(fs_label_list)
data_root = sys_config.data_root
preproc_folder = os.path.join(sys_config.project_root,'data/adni/preprocessed')

# Cost function
age_weight = 0.0   # setting this to zero turns it off
diag_weight = 1.0
weight_decay = 0.00000  # L2 regularization of weights 0.0 turns it off

# Training settings
age_ordinal_regression = True
batch_size = 3
n_accum_batches = 1   # Accumulate the gradients over multiple batches (does not seem to help much).
learning_rate = 0.0001
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = False
momentum = None
bn_momentum = 0.99

# Augmentation settings
augmentation_function = batch_augmentors.flip_augment
do_fliplr = True

# Rarely changed settings
use_data_fraction = False
max_epochs = 20000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
