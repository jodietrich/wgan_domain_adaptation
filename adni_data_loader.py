# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import math

import utils
import image_utils

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

diagnosis_dict = {'CN': 0, 'NC': 0, 'MCI': 1, 'AD': 2}  # NC==CN (it's a bug I accidentally introduced
gender_dict = {'Male': 0, 'Female': 1}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

def fix_nan_and_unknown(target_data_format, input, nan_val=-1, unknown_val=-2):
    if input == 'unknown':
        input = unknown_val
    elif math.isnan(float(input)):
        input = nan_val

    return target_data_format(input)

def crop_or_pad_slice_to_size(image, target_size):

    x_t, y_t, z_t = target_size
    x_s, y_s, z_s = image.shape

    output_volume = np.zeros((x_t, y_t, z_t))

    x_d = abs(x_t - x_s) // 2
    y_d = abs(y_t - y_s) // 2
    z_d = abs(z_t - z_s) // 2

    t_ranges = []
    s_ranges = []

    for t, s, d in zip([x_t, y_t, z_t], [x_s, y_s, z_s], [x_d, y_d, z_d]):

        if t < s:
            t_range = slice(t)
            s_range = slice(d, d + t)
        else:
            t_range = slice(d, d + s)
            s_range = slice(s)

        t_ranges.append(t_range)
        s_ranges.append(s_range)

    # debugging outputs
    # print(target_size)
    # print(image.shape)
    # print(np.asarray(target_size) - np.asarray(image.shape))
    # print('--')

    output_volume[t_ranges[0], t_ranges[1], t_ranges[2]] = image[s_ranges[0], s_ranges[1], s_ranges[2]]



    return output_volume


def prepare_data(input_folder, output_file, size, target_resolution, labels_list, rescale_to_one, image_postfix='.nii.gz'):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    csv_summary_file = os.path.join(input_folder, 'summary_screening.csv')

    summary = pd.read_csv(csv_summary_file)
    summary = summary.loc[summary['image_exists']==True]

    train_and_val_cases, test_cases = train_test_split(summary, test_size=0.2, stratify=summary['diagnosis_3cat'])
    train_cases, val_cases = train_test_split(train_and_val_cases, test_size=0.2, stratify=train_and_val_cases['diagnosis_3cat'])

    hdf5_file = h5py.File(output_file, "w")

    diag_list = {'test': [], 'train': [], 'val': []}
    weight_list = {'test': [], 'train': [], 'val': []}
    age_list = {'test': [], 'train': [], 'val': []}
    gender_list  = {'test': [], 'train': [], 'val': []}
    rid_list = {'test': [], 'train': [], 'val': []}
    confidence_list = {'test': [], 'train': [], 'val': []}
    adas13_list = {'test': [], 'train': [], 'val': []}
    mmse_list = {'test': [], 'train': [], 'val': []}
    field_strength_list = {'test': [], 'train': [], 'val': []}

    file_list = {'test': [], 'train': [], 'val': []}

    logging.info('Counting files and parsing meta data...')

    for train_test, sum_df in zip(['train', 'test', 'val'], [train_cases, test_cases, val_cases]):

        for ii, row in sum_df.iterrows():

            diagnosis_str = row['diagnosis_3cat']
            diagnosis = diagnosis_dict[diagnosis_str]

            if diagnosis not in labels_list:
                continue

            diag_list[train_test].append(diagnosis)

            rid = row['rid']
            rid_list[train_test].append(rid)

            confidence = fix_nan_and_unknown(np.float16, row['confidence'], nan_val=255, unknown_val=254)
            confidence_list[train_test].append(confidence)

            weight_list[train_test].append(row['weight'])
            age_list[train_test].append(row['age'])
            gender_list[train_test].append(gender_dict[row['gender']])
            adas13_list[train_test].append(row['adas13'])
            mmse_list[train_test].append(row['mmse'])

            field_strength = row['field_strength']
            field_strength_list[train_test].append(field_strength)

            phase = row['phase']

            file_name = '%s_%sT_%s_rid%s%s' % (phase.lower(), str(field_strength), diagnosis_str, str(rid).zfill(4), image_postfix)
            file_list[train_test].append(os.path.join(input_folder, file_name))


    # Write the small datasets
    for tt in ['test', 'train', 'val']:

        hdf5_file.create_dataset('rid_%s' % tt, data=np.asarray(rid_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('confidence_%s' % tt, data=np.asarray(confidence_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('age_%s' % tt, data=np.asarray(age_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('gender_%s' % tt, data=np.asarray(gender_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('adas13_%s' % tt, data=np.asarray(adas13_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('mmse_%s' % tt, data=np.asarray(mmse_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('field_strength_%s' % tt, data=np.asarray(field_strength_list[tt], dtype=np.float16))


    n_train = len(file_list['train'])
    n_test = len(file_list['test'])
    n_val = len(file_list['val'])

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'val'], [n_test, n_train, n_val]):
        data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)


    img_list = {'test': [], 'train': [] , 'val': []}

    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'val']:

        write_buffer = 0
        counter_from = 0

        for file in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % file)

            img_dat = utils.load_nii(file)
            img = img_dat[0].copy()

            pixel_size = (img_dat[2].structarr['pixdim'][1],
                          img_dat[2].structarr['pixdim'][2],
                          img_dat[2].structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)


            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1],
                            pixel_size[2] / target_resolution[2]]

            img_scaled = transform.rescale(img,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           multichannel=False,
                                           mode='constant')

            if rescale_to_one:
                img_scaled = image_utils.map_image_to_intensity_range(img_scaled, -1, 1)
            else:
                img_scaled = image_utils.normalise_image(img_scaled)


            img_resized = crop_or_pad_slice_to_size(img_scaled, size)
            img_list[train_test].append(img_resized)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(data, train_test, img_list, counter_from, counter_to)
                _release_tmp_memory(img_list, train_test)

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0



        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, counter_from, counter_to)
        _release_tmp_memory(img_list, train_test)


    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr



def _release_tmp_memory(img_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    gc.collect()


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                label_list,
                                rescale_to_one=False,
                                force_overwrite=False):

    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data
    
    :param input_folder: Folder where the raw ACDC challenge data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    lbl_str = '_'.join([str(i) for i in label_list])

    if rescale_to_one:
        rescale_postfix = '_intrangeone'
    else:
        rescale_postfix = ''

    data_file_name = 'data_size_%s_res_%s_lbl_%s%s.hdf5' % (size_str, res_str, lbl_str, rescale_postfix)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, size, target_resolution, label_list, rescale_to_one=rescale_to_one)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    # input_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI1_screening_reorient_crop'
    input_folder = '/itet-stor/baumgach/bmicdatasets_bmicnas01/Processed/ADNI_Christian/ADNI_ender_selection_no_skullstrip'
    preprocessing_folder = 'preproc_data/ender_wskull'

    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (146, 192, 125), (1.36, 1.36, 1.0), force_overwrite=True)
    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (130, 160, 113), (1.5, 1.5, 1.5), (0,2), force_overwrite=True)
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, (128, 160, 112), (1.5, 1.5, 1.5), (0,2), force_overwrite=True)
