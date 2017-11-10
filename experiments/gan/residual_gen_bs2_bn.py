from experiments.gan.standard_parameters import *

experiment_name = 'residual_identity_gen_bs20_std_disc_10_noise_all_small_data_1e4l1_bn_i1'

# Model settings
residual = True
batch_normalization = True

# regularization settings
w_reg_img_dist_l1 = 1.0e4

# model to use
def generator(z, training, scope_name='generator'):
    return model_zoo.only_conv_generator(z, training, residual=residual, batch_normalization=batch_normalization,
                                         hidden_layers=gen_hidden_layers, filters=gen_filters, input_noise_dim=10, scope_name=scope_name)

discriminator = model_zoo.pool_fc_discriminator_bs2_bn