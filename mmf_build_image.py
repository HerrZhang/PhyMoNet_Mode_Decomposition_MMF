# -*- coding: utf-8 -*-

import numpy as np
# import h5py
import normalization
import mat73
# import scipy.io


def mmf_build_image(num_mode, image_size, num_data, complex_weights):
# =============================================================================
# build light field distribution from the complex weights 
# =============================================================================

    mmf_modes = np.zeros((image_size, image_size, num_mode), dtype=complex)
    
    data = mat73.loadmat('./mmf_data/mmf_'+str(num_mode)+'modes_'+str(image_size)+'.mat')
    
    mmf_modes_temp = data["modes_field"]
    for z in range(0, num_mode):
        mmf_modes[:, :, z] = mmf_modes_temp[z,:,:]
        
   
    # print('Start to generate the mode distribution......\n')
    Image_data = np.zeros(
        (num_data, 1, image_size, image_size))
    Image_data_complex = np.zeros(
        (num_data, 1, image_size, image_size),dtype='complex_')
    for index_number in range(0, num_data):
        single_Image_data = np.zeros(
            (image_size, image_size))
        for mode_index in range(0, num_mode):
            single_Image_data = complex_weights[index_number,mode_index]*mmf_modes[:, :, mode_index]+single_Image_data

        Image_data[index_number, 0, :, :] = normalization.normalization(
            abs(single_Image_data), 0, 1)
        Image_data_complex[index_number, 0, :, :] = single_Image_data

    # print('DONE\n')

    return Image_data,Image_data_complex




def mmf_build_image_single(num_mode, mode_fields, image_size,complex_weights):
# =============================================================================
# build light field distribution from the complex weights 
# =============================================================================
    mmf_modes = np.zeros((image_size, image_size, num_mode), dtype=complex)


    for z in range(0, num_mode):
        mmf_modes[:, :, z] = mode_fields[z,:,:]
        
   
    # print('Start to generate the mode distribution......\n')
    Image_data = np.zeros(
        (image_size, image_size))
    Image_data_complex = np.zeros(
        (image_size, image_size),dtype='complex_')
    single_Image_data = np.zeros((image_size, image_size))
    for mode_index in range(0, num_mode):
        single_Image_data = complex_weights[mode_index]*mmf_modes[:, :, mode_index]+single_Image_data

    Image_data= normalization.normalization(
        abs(single_Image_data), 0, 1)
    Image_data_complex= single_Image_data

    # print('DONE\n')

    return Image_data,Image_data_complex
