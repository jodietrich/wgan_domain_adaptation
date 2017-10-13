import model_zoo
import tensorflow as tf

experiment_name = 'dcgan_fcn_bn_improved_train_fcn_invert_wtest'

# Model settings
model_handle = model_zoo.Std_CNN_bs2_bn

# Data settings
z_dim = 100
data_size = 28*28
data_shape = (28,28)

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
