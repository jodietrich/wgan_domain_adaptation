import model_zoo
import config.system as sys_config
import tensorflow as tf
import os.path
import batch_augmentors

experiment_name = 'joint_genval_gan_bousmalis_gen_n8b4_disc_n8_dropout_keep0.9_no_noise_1e4l1_clfWeight1e5_all_small_final_s15_bs6_i1'

# paths
log_folder = 'joint/final'

# Model settings
clf_model_handle = model_zoo.FCN_multitask_ordinal_bn
multi_task_model = True

# Data settings
image_size = (64, 80, 64)
target_resolution = (1.5, 1.5, 1.5)
offset = (0, 0, -10)
label_list = (0, 2)  # 0 - normal, 1 - mci, 2 - alzheimer's
age_bins = (65, 70, 75, 80, 85)
nlabels = len(label_list)
data_root = sys_config.data_root
preproc_folder = os.path.join(sys_config.project_root, 'data/adni/preprocessed/final')
rescale_to_one = True
use_sigmoid = False
# standard setting
# source_field_strength = 3.0  # magnetic field strength in T of pictures in the source-domain
# target_field_strength = 1.5  # magnetic field strength in T of pictures in the target-domain

# reverse setting
source_field_strength = 1.5  # magnetic field strength in T of pictures in the source-domain
target_field_strength = 3.0  # magnetic field strength in T of pictures in the target-domain

# Cost function
# Interaction of losses (alpha, beta in pixelDA)
# the task loss should be in the same order of magnitude as the Wasserstein GAN loss, so the task loss needs to be scaled up
gan_loss_weight = 1
task_loss_weight = 1e7
# Classifier
age_weight = 0.0
diag_weight = 1.0
weight_decay = 0.0  # 5e-4  #0.00000

# ---------------Classifier--------------

# Training settings
age_ordinal_regression = True
batch_size = 6
n_accum_batches = 1  # currently not implemented
learning_rate_clf = 1e-4
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = False
momentum = None
bn_momentum = 0.99

# Augmentation settings
use_augmentation = False
augmentation_function = None


# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_steps = 1000000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
# be reduced

train_eval_frequency = 500
val_eval_frequency = 100

# -------------- GAN ---------------------
n_channels = 1
image_range = (-1, 1)
# image_range = (-0.512, 2.985) # approximately 1st percentile to 99th percentile of preprocessed ADNI images

# visualization settings
cut_axis = 2  # axis perpendicular to the cut plane (x=0, y=1, z=2)
cut_index = 50  # index of the cut for visualization
diff_threshold = 1  # maximum absolute value for visualization of difference between images. Gets mapped to 255

# Training settings
learning_rate_gan = 1e-4
num_val_batches = 5  # of batches used for validation. Validation happens with a set of size batch_size*num_val_batches

# Improved training settings
improved_training = True
scale = 10.0

# Regularisation settings
w_reg_gen_l1 = 0.0
w_reg_disc_l1 = 0.0
w_reg_gen_l2 = 0.0
w_reg_disc_l2 = 0.0
w_reg_img_dist_l1 = 1.0e4  # weight of l1 distance to source image in gen loss

# Rarely changed settings
max_iterations = 100000
save_frequency = 200
validation_frequency = 100
update_tensorboard_frequency = 10

# Model settings
batch_normalization = True

# noise settings
use_generator_input_noise = False
generator_input_noise_shape = [batch_size, 10]

# model to use
def generator(xs, z_noise, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, z_noise=z_noise, training=training, batch_normalization=batch_normalization,
                                         residual_blocks=4, nfilters=8, scope_reuse=scope_reuse, scope_name=scope_name)

def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    return model_zoo.bousmalis_discriminator(x, training, batch_normalization, middle_layers=5, initial_filters=16,
                                             scope_reuse=scope_reuse, scope_name=scope_name)
