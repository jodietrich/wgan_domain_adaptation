import model_zoo
import config.system as sys_config
import tensorflow as tf
import os.path
import batch_augmentors

experiment_name = 'adni_clf_cropdata_allconv_yesrescale_bs20_all_data_bn_i1'

# Model settings
model_handle = model_zoo.FCN_multitask_ordinal_bn
multi_task_model = True

# Data settings
image_size = (64, 80, 64)
target_resolution =  (1.5, 1.5, 1.5)
offset = (0, 0, -10)
label_list = (0,2)  # 0 - normal, 1 - mci, 2 - alzheimer's
age_bins = (65, 70, 75, 80, 85)
nlabels = len(label_list)
data_root = sys_config.data_root
preproc_folder = os.path.join(sys_config.project_root,'data/adni/preprocessed')
rescale_to_one = True
use_sigmoid = False
source_field_strength = 3.0 # magnetic field strength in T of pictures in the source-domain
target_field_strength = 1.5 # magnetic field strength in T of pictures in the target-domain
training_domain = 'target' # from {'source', 'target', 'all'}. From which domain are the training and validation images.

# Cost function
age_weight = 0.0
diag_weight = 1.0
weight_decay = 0.0  #5e-4  #0.00000

# Training settings
age_ordinal_regression = True
batch_size = 20
n_accum_batches = 1
learning_rate = 1e-4
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = False
momentum = None
bn_momentum = 0.99

# Augmentation settings
augmentation_function = batch_augmentors.flip_augment
do_rotations = False
do_scaleaug = False
do_fliplr = True

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 20000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 500
val_eval_frequency = 100
