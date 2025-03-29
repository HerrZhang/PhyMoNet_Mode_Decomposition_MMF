import numpy as np
import math
# import normalization
import mmf_build_image
# import h5py
import torch

def mmf_data(num_mode,num_data,image_size):
    

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
    [Image,Image_complex] = mmf_build_image.mmf_build_image(
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
