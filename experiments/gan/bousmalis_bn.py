from experiments.gan.standard_parameters import *

experiment_name = 'bousmalis_gen_n16b3_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_i1'

# Model settings
batch_normalization = True

# regularization settings
w_reg_img_dist_l1 = 1.0e4

# model to use
def generator(xs, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, training, batch_normalization, residual_blocks=3, nfilters=16, input_noise_dim=10, scope_reuse=scope_reuse, scope_name=scope_name)

def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    return model_zoo.bousmalis_discriminator(x, training, batch_normalization, middle_layers=5, initial_filters=8, scope_reuse=scope_reuse, scope_name=scope_name)