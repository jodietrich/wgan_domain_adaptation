__author__ = 'jdietric'

from experiments.gan.standard_parameters import *

experiment_name = 'FCN_disc_res_gen_n32b3_all_small_data_0l1_bn_i1'

# Model settings
batch_normalization = True

# model to use
def generator(xs, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, training, batch_normalization, residual_blocks=3, nfilters=32, input_noise_dim=0, scope_reuse=scope_reuse, scope_name=scope_name)

def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    diag_logits= model_zoo.FCN_disc_bn(x, training, nlabels=1, scope_name=scope_name, scope_reuse=scope_reuse)
    return diag_logits