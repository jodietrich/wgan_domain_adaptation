from experiments.gan.standard_parameters import *

experiment_name = 'residual_identity_gen_bs2_std_disc_all_small_data_5e5l1_i1'

# Model settings
residual = True
batch_normalization = False

# regularization settings
w_reg_img_dist_l1 = 5.0e5

# model to use
def generator(z, training, scope_name='generator'):
    return model_zoo.only_conv_generator(z, training, residual=residual, batch_normalization=batch_normalization,
                                         hidden_layers=gen_hidden_layers, filters=gen_filters, scope_name=scope_name)

discriminator = model_zoo.pool_fc_discriminator_bs2