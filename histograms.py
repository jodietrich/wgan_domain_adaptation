import config.system as sys_config
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig


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

def plot_histograms(hist_vectors, fig_name, saving_folder, n_bins='auto', cutoff_left=0.01, show_figure=True):
    # plots the intensity and gradient norm histograms
    fig = plt.figure(fig_name)
    plt.subplot(121)
    plt.hist(hist_vectors['intensity'], bins=n_bins, range=(-1 + cutoff_left, 1))
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')

    plt.subplot(122)
    plt.hist(hist_vectors['gradient_norm'], bins=n_bins, range=(cutoff_left, 2))
    plt.xlabel('gradient norm')
    plt.ylabel('number of pixels')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(saving_folder, fig_name + '.svg')
    print('saving figure as: ' + save_path)
    savefig(save_path, bbox_inches='tight')
    if show_figure:
        plt.show()


if __name__ == '__main__':
    # images path
    plot_name = 'real_1T5_avg'
    img_folder = os.path.join(sys_config.project_root, 'data/generated_images/final/all_experiments')
    saving_folder = '/scratch_net/brossa/jdietric/Documents/thesis/figures/histograms'
    sub_folder = 'source'
    field_strs = ['1.5']
    labels = [0, 2]
    fs_label_combinations = [('1.5', 0), ('1.5', 2)]

    # get all images from the given combinations
    image_folder_path = os.path.join(img_folder, sub_folder)
    file_list = os.listdir(image_folder_path)
    # filter out relevant images
    filtered_file_list = []
    for fs_label_tuple in fs_label_combinations:
        contain_strings = [fs_label_tuple[0] + 'T', 'diag%d' % fs_label_tuple[1]]
        filtered_file_list += [file_name for file_name in file_list if all([str in file_name for str in contain_strings])]

    hist_vectors = {'intensity': [], 'gradient_norm': []}
    for img_name in filtered_file_list:
        img_path = os.path.join(img_folder, sub_folder, img_name)
        # load image
        img_array, _, _ = utils.load_nii(img_path)
        hist_vectors['intensity'].append(make_histogram_vectors(img_array)['intensity'])
        hist_vectors['gradient_norm'].append(make_histogram_vectors(img_array)['gradient_norm'])
    avg_hist_vectors = {'intensity': np.mean(hist_vectors['intensity']), 'gradient_norm': hist_vectors['gradient_norm']}
    plot_histograms(hist_vectors, plot_name, saving_folder)


    # code for single image histogram
     # images path
    # img_folder = os.path.join(sys_config.project_root, 'data/generated_images/final/all_experiments')
    # sub_folder = 'residual_gen_n8b4_disc_n8_bn_dropout_keep0.9_10_noise_all_small_data_1e4l1_s3_final_i1'
    # img_num = 468
    # field_str = '1.5'
    # label = 2
    # img_name = 'generated_img_%sT_diag%d_ind%d.nii.gz' % (field_str, label, img_num)
    # saving_folder = '/scratch_net/brossa/jdietric/Documents/thesis/figures/histograms'
    # field_str = field_str.replace('.', '')
    # plot_name = 'separate_residual_no_noise_target' + field_str + 'T_' + str(img_num)
    # img_path = os.path.join(img_folder, sub_folder, img_name)
    # # load image
    # img_array, _, _ = utils.load_nii(img_path)
    # hist_vectors = make_histogram_vectors(img_array)
    # plot_histograms(hist_vectors, plot_name, saving_folder)