from experiments.gan.standard_parameters import *

experiment_name = 'std_cnn_identity_gen_bs2_bn_i1'

# Model settings
residual = False
batch_normalization = True

# model to use
def generator(z, training, scope_name='generator'):
    return model_zoo.only_conv_generator(z, training, residual=residual, batch_normalization=batch_normalization,
                                         scope_name=scope_name, hidden_layers=gen_hidden_layers, filters=gen_filters)

discriminator = model_zoo.pool_fc_discriminator_bs2_bn