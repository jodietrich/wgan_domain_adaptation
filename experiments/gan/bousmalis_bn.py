from experiments.gan.standard_parameters import *

experiment_name = 'bousmalis_bn_dropout_keep0.9_i1'

# Model settings
batch_normalization = True

# model to use
def generator(xs, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, training, batch_normalization, residual_blocks=4, nfilters=8, scope_reuse=scope_reuse, scope_name=scope_name)

def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    return model_zoo.bousmalis_discriminator(x, training, batch_normalization, middle_layers=5, initial_filters=8, scope_reuse=scope_reuse, scope_name=scope_name)