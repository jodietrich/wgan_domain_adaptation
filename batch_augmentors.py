import numpy as np

import adni_data_loader
from batch_generator_list import iterate_minibatches


def flip_augment(X, y_list, do_fliplr=True):

    N = X.shape[0]

    X_list = []

    for ii in range(N):

        img = np.squeeze(X[ii,...])

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.flip(img, 0)


        X_list.append(img[..., np.newaxis])

    X_ = np.asarray(X_list)

    return X_, y_list


if __name__ == '__main__':

    from experiments import jia_xi_net as exp_config
    import matplotlib.pyplot as plt

    data = adni_data_loader.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        label_list=exp_config.fs_label_list,
        force_overwrite=False
    )

    for batch in iterate_minibatches(data['images_train'],
                                     [data['diagnosis_train'], data['age_train']],
                                     batch_size=exp_config.batch_size,
                                     augmentation_function=None,  #flip_augment,
                                     exp_config=exp_config):

        X, [y, a] = batch

        X_, [y_, a_] = flip_augment(X, [y, a], exp_config.do_fliplr)

        fig1 = plt.figure()
        fig1.add_subplot(131)
        plt.imshow(np.squeeze(X[0,80,:,:]))
        fig1.add_subplot(132)
        plt.imshow(np.squeeze(X[0,:,80,:]))
        fig1.add_subplot(133)
        plt.imshow(np.squeeze(X[0,:,:,80]))

        fig2 = plt.figure()
        fig2.add_subplot(131)
        plt.imshow(np.squeeze(X_[0,80,:,:]))
        fig2.add_subplot(132)
        plt.imshow(np.squeeze(X_[0,:,80,:]))
        fig2.add_subplot(133)
        plt.imshow(np.squeeze(X_[0,:,:,80]))

        plt.show()