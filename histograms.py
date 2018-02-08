import config.system as sys_config
import os.path
import utils
import numpy as np
import matplotlib.pyplot as plt


# make histograms to compare 1.5 T and 3 T images and generated images
# histograms of intensity and intensity gradient with respect to the spacial dimensions


def make_histogram_vectors(image):
    # image must be a numpy array
    vectors = {}
    vectors['intensity'] = image.flatten()
    vectors['gradient_norm'] = pixel_gradient_norm_list(image)
    return vectors

def pixel_gradient_norm_list(image):
    difference_images = pixel_difference_gradients(image)
    pixel_gradient_norms = np.linalg.norm(difference_images, ord=2, axis=-1)
    pixel_gradient_norm_vector = pixel_gradient_norms.flatten()
    return pixel_gradient_norm_vector


def pixel_difference_gradients(image):
    pixel_dif1 = image[1:, :-1, :-1] - image[:-1, :-1, :-1]
    pixel_dif2 = image[:-1, 1:, :-1] - image[:-1, :-1, :-1]
    pixel_dif3 = image[:-1, :-1, 1:] - image[:-1, :-1, :-1]
    return np.stack((pixel_dif1, pixel_dif2, pixel_dif3), axis=-1)

def plot_histograms(hist_vectors, plot_title, n_bins='auto'):
    # plots the intensity and gradient norm histograms
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(hist_vectors['intensity'], bins=n_bins)
    axs[1].hist(hist_vectors['gradient_norm'], bins=n_bins)
    plt.show()
    # TODO: save plot


if __name__ == '__main__':
    # images path
    img_folder = os.path.join(sys_config.project_root, 'data/generated_images/final/all_experiments')
    sub_folder = 'source'
    img_name = 'source_img_1.5T_0.nii.gz'
    img_path = os.path.join(img_folder, sub_folder, img_name)
    # load image
    img_array, _, _ = utils.load_nii(img_path)
    hist_vectors = make_histogram_vectors(img_array)
    plot_histograms(hist_vectors, img_name)