import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def iterate_minibatches(images,
                        labels_list,
                        batch_size,
                        exp_config,
                        selection_indices=None,
                        augmentation_function=None,
                        map_labels_to_standard_range=True,
                        shuffle_data=True):
    '''
    Function to create mini batches from the dataset of a certain batch size
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param selection_indices: indices from which images are selected. If this is None the selection is from all images
    :param augment_batch: should batch be augmented?
    :return: mini batches
    '''
    if selection_indices is None:
        random_indices = np.arange(images.shape[0])
    else:
        random_indices = selection_indices
    if shuffle_data:
        np.random.shuffle(random_indices)

    n_images = len(random_indices)

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        # y = labels[batch_indices, ...]

        y_list = [y_ll[batch_indices,...] for y_ll in labels_list]

        # DEBUG
        # print(y_list)

        if map_labels_to_standard_range:
            # This puts the labels in a range from 0 to nlabels.
            # E.g. [0,0,2,2] becomes [0,0,1,1] (if 1 doesnt exist in the data)
            y_list[0] = np.asarray([np.argwhere(i==np.asarray(exp_config.label_list)) for i in y_list[0]]).flatten()

        image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)

        if augmentation_function:
            X, y_list = augmentation_function(X, y_list, do_fliplr=exp_config.do_fliplr)


        yield X, y_list




if __name__ == '__main__':

    import utils

    images = np.zeros((200,10,10,15))
    labels = [np.random.random_integers(0, 2, 200), np.random.random_integers(5, 8, 200)]
    exp_config = utils.Bunch(image_size=(10, 10, 15), do_rotations=False, do_scaleaug=False, do_fliplr=False, label_list=(0,1,2))

    for batch in iterate_minibatches(images, labels, 3, exp_config):
        x, [y1, y2] = batch
        # print(x)
        print(y1)
        print(y2)
        print('--')


    # for batch in iterate_minibatches_stratified(images, labels, [2,2,3,1], exp_config):
    #     x, y = batch
    #     # print(x)
    #     print(y)
    #     print('--')

    # for batch in iterate_minibatches_stratified_iteration_based(images, labels, [2,2,3,1], exp_config, num_iterations=10):
    #     x, y = batch
    #     # print(x)
    #     print(y)
    #     print('--')
    #
    # iter = iterate_minibatches_stratified_iteration_based(images, labels, [2, 2, 3, 1], exp_config, num_iterations=10)
    # x, y = iter.__next__()
    # print(y)
    # x, y = iter.__next__()
    # print(y)
