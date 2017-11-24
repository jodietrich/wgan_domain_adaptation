from experiments.gan.standard_parameters import *

experiment_name = 'bousmalis_gen_n8b4_disc_n8_bn_dropout_keep0.9_no_noise_all_small_data_1e4l1_s3_final_i1'

# Model settings
batch_normalization = True

# regularization settings
w_reg_img_dist_l1 = 1.0e4

# noise settings
use_generator_input_noise = False


# model to use
def generator(xs, z_noise, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, z_noise=z_noise, training=training, batch_normalization=batch_normalization,
                                         residual_blocks=4, nfilters=8, scope_reuse=scope_reuse, scope_name=scope_name)


def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    return model_zoo.bousmalis_discriminator(x, training, batch_normalization, middle_layers=5, initial_filters=8,
                                             scope_reuse=scope_reuse, scope_name=scope_name)
