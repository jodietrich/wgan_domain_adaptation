# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import nibabel as nib
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import config.system as sys_config
import logging
import tensorflow as tf

def fstr_to_label(fieldstrengths, field_strength_list, label_list):
    # input fieldstrenghts hdf5 list
    # field_strength_list must have the same size as label_list
    # returns a numpy array of labels
    assert len(label_list) == len(field_strength_list)
    labels = np.empty_like(fieldstrengths, dtype=np.int16)
    for fs_ind, current_field_strength in enumerate(fieldstrengths):
        valid_value = False
        for label_ind, current_label in enumerate(label_list):
            if(current_field_strength == field_strength_list[label_ind]):
                labels[fs_ind] = current_label
                valid_value = True
                break
        if(not valid_value):
            raise ValueError('unexpected value in fieldstrengths: %s' % current_field_strength)
    return labels



def age_to_ordinal_reg_format(ages, bins=(65, 70, 75, 80, 85)):

    N = ages.shape[0]
    P = len(bins)

    ages_mat = np.transpose(np.tile(ages,(P,1)))
    bins_mat = np.tile(bins, (N,1))

    return np.array(ages_mat>bins_mat, dtype=np.uint8)

def age_to_bins(ages,  bins=(65, 70, 75, 80, 85)):

    ages_ordinal = age_to_ordinal_reg_format(ages, bins)
    return np.sum(ages_ordinal, axis=-1)


def ordinal_regression_to_bin(ages_ord_reg):

    # N = ages_ord_reg.shape[0]
    # binned_list = []
    # for nn in range(N):
    #     if np.sum(ages_ord_reg[nn,:]) > 0:
    #         binned_list.append(all_argmax(ages_ord_reg[nn,:])[-1][0]+1)
    #     else:
    #         binned_list.append(0)



    return np.sum(ages_ord_reg, -1)


def get_ordinal_reg_weights(ages_ordinal_reg):

    ages_binned = list(ordinal_regression_to_bin(ages_ordinal_reg))
    P = ages_ordinal_reg.shape[1]

    counts = [ages_binned.count(pp) for pp in range(P)]
    counts = [np.divide(np.sqrt(cc), np.sum(np.sqrt(counts))) for cc in counts]

    return counts

def all_argmax(arr, axis=None):

    return np.argwhere(arr == np.amax(arr, axis=axis))


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def create_and_save_nii(data, img_path):

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, img_path)

def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))


def index_sets_to_selectors(*index_sets):
    # takes in sets of indices and changes them to lists with True if the index was in the set and false otherwise
    # works with lists or tuples of indices as well, but the in operation is O(n) instead of O(1)
    selector_result = []
    for ind_set in index_sets:
        selector_result.append([(index in ind_set) for index in range(max(ind_set))])
    return selector_result


# Useful shortcut for making struct like contructs
# Example:
# mystruct = Bunch(a=1, b=2)
# print(mystruct.a)
# >>> 1

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def load_log_exp_config(experiment_path, file_name=None, other_py_files=['standard_parameters.py']):
    # loads the module of the experiment and returns a loader that can be used to access variables and classes in the
    # module (loader.myClass())
    # if the file_name of the module is not given then the file of the module must be the only .py file in the directory
    # except for the files in other_py_files

    if file_name is None:
        # get experiment config file (assuming it is the first python file in log directory)
        py_file_list = [file for file in os.listdir(experiment_path) if (file.endswith('.py') and file not in other_py_files)]

        if len(py_file_list) != 1:
            raise ValueError('unexpected py files in log directory or experiment file not found')
        py_file_name = py_file_list[0]
    else:
        py_file_name = file_name
    py_file_path = os.path.join(experiment_path, py_file_name)

    # import config file
    # remove the .py with [:-3]
    experiment_module = SourceFileLoader(py_file_name[:-3], py_file_path).load_module()

    # experiment name is the same as the folder name
    experiment_folder_name = experiment_path.split('/')[-1]
    if experiment_folder_name != experiment_module.experiment_name:
        logging.warning('warning: the experiment folder name %s is different from the experiment name %s'
                        % (experiment_folder_name, experiment_module.experiment_name))

    return experiment_module, experiment_path

def string_dict_in_order(dict, key_function=None, key_string='', value_string=''):
    # key is a function to give the elements in the dictionary a numerical value that is used for the order
    separator = '\n'
    lines = []
    for dict_key in sorted(dict, key=key_function, reverse=True):
        lines.append(key_string + str(dict_key) + ' ' + value_string + str(dict[dict_key]))
    print_string = separator.join(lines)
    return print_string

def module_from_path(path):
    module_name = os.path.splitext(os.path.split(path)[1])[0]
    return SourceFileLoader(module_name, path).load_module()

def get_latest_checkpoint_and_step(logdir, filename):
    init_checkpoint_path = get_latest_model_checkpoint_path(logdir, filename)
    logging.info('Checkpoint path: %s' % init_checkpoint_path)
    last_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1])
    logging.info('Latest step was: %d' % last_step)
    return init_checkpoint_path, last_step

def get_session_memory_config():
    # prevents ResourceExhaustError when a lot of memory is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not defined in the default device, let it execute in another.
    return config





