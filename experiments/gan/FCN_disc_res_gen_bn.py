__author__ = 'jdietric'

from experiments.gan.standard_parameters import *

experiment_name = 'FCN_disc_res_gen_n16b3_no_noise_all_small_data_1e4l1_bn_i1'

# Model settings
batch_normalization = True


# model to use
def generator(xs, z_noise, training, scope_reuse=False, scope_name='generator'):
    return model_zoo.bousmalis_generator(xs, z_noise=z_noise, training=training, batch_normalization=batch_normalization,
                                         residual_blocks=3, nfilters=16, scope_reuse=scope_reuse, scope_name=scope_name)


def discriminator(x, training, scope_reuse=False, scope_name='discriminator'):
    diag_logits = model_zoo.FCN_disc_bn(x, training, nlabels=1, scope_name=scope_name, scope_reuse=scope_reuse)
    return diag_logits
