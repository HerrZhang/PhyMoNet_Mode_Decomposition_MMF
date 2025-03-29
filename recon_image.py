#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math
from CorrCoef import corr2,corr2_tensor
import torch


def recon_prediction(num_modes, num_test_data, pred, phase_variants, image_test, mode_fields):
    num_modes = int(num_modes)

    weights_complex = np.zeros((num_test_data, num_modes), dtype='complex')
    corr_N = np.zeros([num_test_data, 1])

    time_begin = time.time()
    # np.corrcoef(x,y)
    for i in range(num_test_data):


        ground_truth_i = image_test[i, 0, :, :]
       

        corr_N[i] = corr2(ground_truth_i, abs(pred))
        weights_complex[i, :] = 1

    print('mean corr: {:.4f}%'.format(corr_N.mean()*100))

    time_elapsed = time.time() - time_begin
    # print('Elapsed {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    return corr_N, weights_complex


def recon_prediction_tensor(num_modes, device, num_test_data, pred, phase_variants, image_test, mode_fields):
    num_modes = int(num_modes)
    phase_variants_tensor = torch.from_numpy(phase_variants).to(device)
    image_test_tensor = image_test.to(device)
    mode_fields_tensor = torch.from_numpy(mode_fields).to(device)

    weights_amp_tensor = pred[:, 0:num_modes]

    weights_phase = pred[:, num_modes:(2*num_modes-1)]

    weights_phase[weights_phase > 1] = 1
    weights_phase[weights_phase < 0] = 0
    # print("phase values have been set to [0,1].")
    weights_phase_n = weights_phase*2 + (-1)

    weights_phase = torch.arccos(weights_phase_n)
    # print("number of NaN")
    # print(len(np.argwhere(np.isnan(weights_phase)))/2)

    first_column = torch.zeros([num_test_data, 1]).to(device)
    weights_phase_0 = torch.cat([first_column, weights_phase], dim=1)

    weights_complex = torch.from_numpy(
        np.zeros((num_test_data, num_modes), dtype='complex'))
    corr_N = torch.zeros([num_test_data, 1]).to(device)

    time_begin = time.time()
    # np.corrcoef(x,y)
    for i in range(num_test_data):

        phase_vectors = weights_phase_0[i, :] * phase_variants_tensor
        complex_vectors = torch.full_like(phase_vectors, 0, dtype=torch.cfloat)

        ground_truth_i = image_test_tensor[i, 0, :, :]
        corr_n = torch.zeros(phase_vectors.shape[0])
        for i_phase in range(phase_vectors.shape[0]):

            complex_vector = weights_amp_tensor[i, :] * \
                torch.pow(math.e, phase_vectors[i_phase, :]*1j)
            # abs(complex_vector)
            # np.angle(complex_vector)
            image_recon = torch.full_like(ground_truth_i, 0).to(device)
            # mode combination
            for index_mode in range(num_modes):
                image_recon = image_recon + \
                    complex_vector[index_mode] * \
                    mode_fields_tensor[index_mode, :, :]

            corr_n[i_phase] = corr2_tensor(ground_truth_i, abs(image_recon))
            complex_vectors[i_phase, :] = complex_vector
        # find the maximal corr, if there are 2 elements, only use one of them

        max_index = np.where(corr_n == corr_n.min())
        # print(max_index)
        if max_index[0].size > 1:
            max_index = max_index[0][0]

        corr_N[i] = corr_n[max_index]
        weights_complex[i, :] = complex_vectors[max_index, :]

    print('mean corr: {:.4f}%'.format(corr_N.mean()*100))

    time_elapsed = time.time() - time_begin
    # print('Elapsed {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    return corr_N, weights_complex


def recon_vectors(weights_complex, mode_fields):
    num_modes = mode_fields.shape[0]
    num_images = weights_complex.shape[0]
    image_size = mode_fields.shape[1]
    complex_fields = np.zeros(
        [num_images, 1, mode_fields.shape[1], mode_fields.shape[2]], dtype='complex')
    for i in range(num_images):
        image_recon = np.zeros([mode_fields.shape[1], mode_fields.shape[2]])

        for index_mode in range(num_modes):
            image_recon = image_recon + \
                weights_complex[i, index_mode]*mode_fields[index_mode, :, :]

        complex_fields[i, 0, :, :] = image_recon

    return complex_fields
