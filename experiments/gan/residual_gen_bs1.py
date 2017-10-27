from experiments.gan.standard_parameters import *

experiment_name = 'residual_identity_gen_bs1_std_disc_i1'

# Model settings
residual = True
batch_normalization = False
gen_hidden_layers = 3
gen_filters = 32

# Training settings
batch_size = 1
num_val_batches = 20 # must be smaller than the smallest validation set

# model to use
def generator(z, training, scope_name='generator'):
    return model_zoo.only_conv_generator(z, training, residual=residual, batch_normalization=batch_normalization,
                                         hidden_layers=gen_hidden_layers, filters=gen_filters, scope_name=scope_name)

discriminator = model_zoo.pool_fc_discriminator_bs1