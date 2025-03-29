# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:44:59 2023

@author: sy
"""
import numpy as np
import mmf_build_image
import matplotlib.pyplot as plt
import normalization
import math


def rescal_result(x,num_modes,image_size):
    
    amp = x[0,0:num_modes].reshape(1, num_modes)
    phase = x[0,num_modes:]
    amp_n = amp*amp
    # amp_n = normalization.normalization(amp, 0, 1)

    amp_n = amp_n/ np.linalg.norm(amp_n) 
    # amp_n = 
    # amp_n[amp_n < 0] = 0
    for index, item in enumerate(phase):
        if item >= 0:
            phase[index]  = phase[index]  % (2*math.pi)
        if item < 0 :
            while phase[index]  < 0 :
                phase[index]   = phase[index]+(2*math.pi)
    
    # amp_n = normalization.normalization(amp, 0, 1)
    # phase_n = normalization.normalization(phase, -math.pi, math.pi)
    
    rel_phs = np.zeros((1,num_modes))
    for mode_index0 in range(0,num_modes-1):
        # rel_phs[0,mode_index0+1] = phase_n[mode_index0]
        rel_phs[0,mode_index0+1] = phase[mode_index0]
    complex_weights_vector = np.concatenate((amp_n, rel_phs), axis=0)
    k_i = amp_n*np.exp(1j*rel_phs) 
    Image_data,Image_data2 = mmf_build_image.mmf_build_image(num_modes, image_size, np.size(k_i, 0), k_i)

    
    return complex_weights_vector,Image_data,Image_data2,k_i