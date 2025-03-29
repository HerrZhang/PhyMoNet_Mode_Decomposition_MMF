#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:35:57 2023

@author: qzhang
"""

# -*- coding: utf-8 -*-

# import h5py
import normalization
import mat73
# import scipy.io
import numpy as np
import math
# import normalization
# import h5py
import torch

#%% build image
def mmf_build_image(num_mode, image_size, num_data, complex_weights):
# =============================================================================
# build light field distribution from the complex weights 
# =============================================================================

    mmf_modes = np.zeros((image_size, image_size, num_mode), dtype=complex)
    # print(num_mode)
    # time_begin = time.time()
    data = mat73.loadmat('./mmf_mat_data/mmf_'+str(num_mode)+'modes_'+str(image_size)+'.mat')
    # time.sleep(1)
    # time_elapsed = time.time() - time_begin
    
    # print('Elapsed {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
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



#%% build single image
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


#%%data generation
def mmf_data_generation(num_mode,num_data,image_size):
    

    # =============================================================================
    # generate random mode weights combinations
    # =============================================================================
    
    # amplitude weights: number of data x N 
    amp = np.random.rand(num_data, num_mode)    
    # amplitude weights after normaliaztion
    amp_n = np.zeros((num_data, num_mode)) 
    
    # phase weights: numer of data x N (first mode = 0 )
    # phase range: [-pi, pi]
    # phase = (np.random.rand(num_data, (num_mode-1))*2-1)*math.pi
    # phase range: [0, 2pi]
    phase = np.random.rand(num_data, (num_mode-1))*2*math.pi
    first_phase = np.zeros((num_data, 1))
    # calculate the relative phase difference
    rel_phs = np.hstack((first_phase,phase))
    
    # mode weights in complex value: number of data x N 
    complex_weights = np.zeros((num_data, num_mode), dtype='complex')
    # mode weights in vector form (2N-1)
    weights_vectors = np.zeros((num_data, 2*num_mode-1))
    
    # define 
    # data_set = np.zeros((num_data, 2*num_mode-1),dtype = np.float32)
    
    for i in range(0, num_data):
        # calculate the norm of each data 
        amp_n[i, :] = amp[i, :] / np.linalg.norm(amp[i, :])
        # mode weights in complex value (1xN)
        complex_weights[i, :] = amp_n[i, :]*np.exp(1j*rel_phs[i, :])
        
        # mode weights in vector form (2N-1): phase for 1st mode = 0
        weights_vectors[i,:] = np.hstack((amp_n[i,:], phase[i, :]))
        
    # =============================================================================
    # generate complex light field distributions and amplitude distributions
    # =============================================================================
    # superposition of modes for light field generation
    # image_data: only amplitude distribution
    # image_data_complex: complex distribution 
    print('Start to generate the mode distribution......\n')
    # print(num_mode)
    [Image,Image_complex] = mmf_build_image(
        num_mode, image_size, num_data, complex_weights)
    print('DONE\n')
    # =============================================================================
    # # tranfer np data to tensor data
    # =============================================================================
    # only amplitude distribution for training
    Image_tensor = torch.tensor(Image, dtype=torch.float32)
    # complex field distribution
    Image_complex_tensor = torch.tensor(Image_complex, dtype=torch.complex64)

    # label vector 
    weights_vectors_tensor = torch.tensor(weights_vectors, dtype=torch.float32)
    # mode weight in complex format
    complex_weights_tensor = torch.tensor(complex_weights, dtype=torch.complex64)
    
    return Image_tensor,weights_vectors_tensor,Image_complex_tensor,complex_weights_tensor

def mmf_data_generation_limitedPhase(num_mode,num_data,image_size):
    

    # =============================================================================
    # generate random mode weights combinations
    # =============================================================================
    
    # amplitude weights: number of data x N 
    amp = np.random.rand(num_data, num_mode)    
    # amplitude weights after normaliaztion
    amp_n = np.zeros((num_data, num_mode)) 
    
    # phase weights: numer of data x N (first mode = 0 )
    # phase range: [-pi, pi]
    # phase = (np.random.rand(num_data, (num_mode-1))*2-1)*math.pi
    # phase range: [0, 2pi]
    phase_rest = np.random.rand(num_data, (num_mode-2))*2*math.pi
    
    phase_second = np.random.rand(num_data, 1)*math.pi
    phase = np.hstack((phase_second,phase_rest))

    first_phase = np.zeros((num_data, 1))
    # calculate the relative phase difference
    rel_phs = np.hstack((first_phase,phase))
    
    # mode weights in complex value: number of data x N 
    complex_weights = np.zeros((num_data, num_mode), dtype='complex_')
    # mode weights in vector form (2N-1)
    weights_vectors = np.zeros((num_data, 2*num_mode-1))
    
    # define 
    # data_set = np.zeros((num_data, 2*num_mode-1),dtype = np.float32)
    
    for i in range(0, num_data):
        # calculate the norm of each data 
        amp_n[i, :] = amp[i, :] / np.linalg.norm(amp[i, :])
        # mode weights in complex value (1xN)
        complex_weights[i, :] = amp_n[i, :]*np.exp(1j*rel_phs[i, :])
        
        # mode weights in vector form (2N-1): phase for 1st mode = 0
        weights_vectors[i,:] = np.hstack((amp_n[i,:], phase[i, :]))
        
    # =============================================================================
    # generate complex light field distributions and amplitude distributions
    # =============================================================================
    # superposition of modes for light field generation
    # image_data: only amplitude distribution
    # image_data_complex: complex distribution 
    print('Start to generate the mode distribution......\n')
    # print(num_mode)
    [Image,Image_complex] = mmf_build_image(
        num_mode, image_size, num_data, complex_weights)
    print('DONE\n')
    # =============================================================================
    # # tranfer np data to tensor data
    # =============================================================================
    # only amplitude distribution for training
    Image_tensor = torch.tensor(Image, dtype=torch.float32)
    # complex field distribution
    Image_complex_tensor = torch.tensor(Image_complex, dtype=torch.complex64)

    # label vector 
    weights_vectors_tensor = torch.tensor(weights_vectors, dtype=torch.float32)
    # mode weight in complex format
    complex_weights_tensor = torch.tensor(complex_weights, dtype=torch.complex64)
    
    return Image_tensor,weights_vectors_tensor,Image_complex_tensor,complex_weights_tensor

