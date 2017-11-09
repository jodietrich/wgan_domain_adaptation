from experiments.gan.standard_parameters import *

experiment_name = 'bousmalis_bn_dropout_keep0.9_no_noise_all_small_data_1e5l1_i1'

# Model settings
batch_normalization = True

# regularization settings
w_reg_img_dist_l1 = 1.0e5

# model to use
def generator(xs, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, training, batch_normalization, residual_blocks=4, nfilters=8, input_noise_dim=0, scope_reuse=scope_reuse, scope_name=scope_name)

def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    return model_zoo.bousmalis_discriminator(x, training, batch_normalization, middle_layers=5, initial_filters=8, scope_reuse=scope_reuse, scope_name=scope_name)