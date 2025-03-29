#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:38:37 2023

@author: qzhang

"""
#%% Import modules
import numpy as np
import math
import normalization
# from mmf_build_image import mmf_build_image,mmf_build_image_single
from mmf_functions import mmf_data_generation

import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import torch
import mat73
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from scipy.io import savemat
from models import MLP
from models import VGG
from CorrCoef import corr2
# from model_function import train
from PhyMoNet_function import train_PhyMoNet

# from PhyMoNet_function import train_phymonet
from PhyMoNet_function import test_PhyMoNet
from PhyMoNet_function import rescal_result_PhyMoNet
from PhyMoNet_function import analyses_results

model_save_folder = '/results/'
training_save_folder = './results/'

#%% 
def create_model(model_type_index,mode_fields):
    
    model_types = ["MLP", "VGG", "DenseNet", "MTNet"]
    model_type = model_types[model_type_index]
    print("Train "+model_type+" model")
    if model_type == "MLP":
        model = MLP.MLP_MD(num_modes, image_size,mode_fields)

    # if model_type == "VGG":
        # model = VGG.VGG16_MD(num_modes)

    return model

#%%  for loop 

# num_mode_option = [19] #43
num_mode_option = [23, 40] 
seed = 92

for index_num in range(len(num_mode_option)):
    num_modes=num_mode_option[index_num]
    #%% define the data parameters

    num_data = 1
    image_size = 64 #64
            
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # image data in tensor-format
    # image: amplitude distribution (C x 1 x H x W)
    # weights_vectors: amp + phase (C x 2N-1)
    # image_complex: complex distribution (C x 1 x H x W)
    # complex_weights: complex number mode weights (C x N)
    [Images,weights_vectors,Image_complex,complex_weights] = mmf_data_generation(num_modes,num_data,image_size)
    
    mode_fields = np.zeros((num_modes,image_size,image_size),dtype="complex_")
    
    mode_fields_mat = mat73.loadmat('./mmf_mat_data/mmf_'+str(num_modes)+'modes_'+str(image_size)+'.mat')
    temp_mode_fields = mode_fields_mat["modes_field"]  
    # print(temp_mode_fields.shape)
    for index_mode in range(num_modes):
        mode_fields[index_mode,:,:] = temp_mode_fields[index_mode,:,:]
                        
    #%% Hyperparameters for training 
    # =============================================================================
    # learning rate .... 
    # create a model for mode decomposition 
    # =============================================================================
    # batch_size_option = ([1])
    flag_validation = 0
    flag_test = 1
    test_fr = 1  # test frequency
    
    batch_size = 1
    lr_option = np.array([0.0001])
    lr = 0.0001
    num_epochs = 1000
    loss_threshold = 0.00001
    cuda_index = 'cuda:0' # using GPU
    cuda_index = 'cpu' # using CPU

    lr_step = 200
    lr_ratio = 0.9
    
    criterion = nn.L1Loss() 
    
    #%% define a neural network
  
    model_type_index = 0  # 0:MLP, 1:VGG, 2:DenseNet, 3:MTNet
    
    mmf_modes = np.zeros((image_size, image_size, num_modes), dtype=complex)
    for z in range(0, num_modes):
        mmf_modes[:, :, z] = mode_fields[z,:,:]
        
    model = create_model(model_type_index,mmf_modes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model_types = ["MLP", "VGG"]
    model_type = model_types[model_type_index]
    
    #%% training
    # Input_image: Imgae
    # Ouput_label: weights_vectors
    
    # correlation coefficient 
    cc_N = np.zeros([num_data])
    cc_N_corr = np.zeros([num_data]) # based on corrected phase weigths
    # predicted vectors 
    pre_vectors_ori = np.zeros([num_data,(2*num_modes-1)])
    pre_vectors_rescale = np.zeros([num_data,(2*num_modes-1)])
    pre_vectors_rescale_comp = np.zeros([num_data,num_modes],dtype="complex_")
    
    # predicted images
    pre_Images = np.zeros([num_data,image_size,image_size])
    pre_Images_comp = np.zeros([num_data,image_size,image_size],dtype="complex_")
    pre_Images_rescale = np.zeros([num_data,image_size,image_size])
    pre_Images_rescale_comp = np.zeros([num_data,image_size,image_size],dtype="complex_")
    
    # weights error 
    amp_errors = np.zeros([num_data,num_modes])
    pha_errors = np.zeros([num_data,num_modes-1])
    pha_errors_corr = np.zeros([num_data,num_modes-1])
    
    # training process 
    training_cc_N = []
    training_cc_index_N = []
    Loss_epoch_N = []
    
    device = torch.device(cuda_index)
    train_dataset = torch.utils.data.TensorDataset(Images, weights_vectors)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index_data, (Image, Label) in enumerate(train_loader,0):
        print("mode number:")
        print(num_modes)
        print("data index")
        print(index_data)
    
        model = create_model(model_type_index,mmf_modes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        [activation,DNN_outputs,training_cc_index,training_cc,Loss_epoch] = train_PhyMoNet(model,optimizer,criterion,cuda_index, 
                        Image,Label,loss_threshold,
                        num_epochs,lr_step, lr_ratio, 
                        flag_test, test_fr,mode_fields)
        
        training_cc_N.append(training_cc)
        training_cc_index_N.append(training_cc_index)
        Loss_epoch_N.append(Loss_epoch)    
        
        pre_Images[index_data,:,:] = abs(DNN_outputs.cpu().detach().numpy())
        
        pre_Images_comp[index_data,:,:] = DNN_outputs.cpu().detach().numpy()
        # original predicted vector
        pre_vectors_ori[index_data,:] = activation
        cc_N[index_data] = test_PhyMoNet(model,Image,Label,device,mode_fields)  
        print('Correlation Coefficient:')
        print(cc_N[index_data] )
        pred_vector_rescale,pre_Image,pre_Image_complex,pre_complex_weights= rescal_result_PhyMoNet(activation, num_modes, image_size)
        pre_Images_rescale[index_data,:,:] = pre_Image
        pre_Images_rescale_comp[index_data,:,:] = pre_Image_complex
        
        # cc_N = test_PhyMoNet_dataloader(model,train_loader,device,phase_variants,mode_fields)  
        pre_vectors_rescale[index_data,:] = pred_vector_rescale
        pre_vectors_rescale_comp[index_data,:] = pre_complex_weights
        
       
    #%% save
    if 0:
        save_training_name= training_save_folder + str(num_modes)+"modes_"+model_type+'_Loss_'+str(image_size)+'_'+str(num_epochs)+'_lr_'+str(lr)+'_testnum_'+str(num_data)
        
        savemat(save_training_name + '.mat',
                { "Loss_epoch": Loss_epoch_N, 
                  "num_epochs": num_epochs,
                  "training_cc_N": training_cc_N,
                  "training_cc_index_N": training_cc_index_N,
                  "l_r": lr,
                  "Test_Images": Images.numpy(),
                  "Test_Images_comp": Image_complex.numpy(),
                  "Test_Labels": weights_vectors.numpy(),
                  
                  "Pre_Images": pre_Images,
                  "Pre_Images_complex": pre_Images_comp,
                  "Pre_Images_rescale": pre_Images_rescale,
                  "Pre_Images_rescale_complex": pre_Images_rescale_comp,
                  
                  "Pre_Vectors": pre_vectors_ori,
                  "Pre_Vectors_rescale": pre_vectors_rescale,
                  "Pre_Vectors_rescale_complex": pre_vectors_rescale_comp,
                  "cc_N": cc_N
                  })
        
                
        print('The training process has been save in:')
        print(save_training_name)
        
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    #%% calculate errors
    amplitude_weights_true = np.transpose(weights_vectors[0,0:num_modes].numpy())
    phase_weights_true = weights_vectors[0,num_modes-1:2*num_modes+1].numpy()-np.pi
    phase_weights_true[0] = 0
    
    amplitude_weights_pre = np.abs(pre_vectors_rescale_comp[0,0:num_modes])
    phase_weights_pre = np.angle(pre_vectors_rescale_comp[0,:])
    
    amplitude_weights_e = abs(amplitude_weights_true-amplitude_weights_pre).mean()
    phase_weights_e = abs(phase_weights_true-phase_weights_pre).mean()/(2 * np.pi)
    print("amplitude weights error: {:.4e}".format(amplitude_weights_e))
    print("phase weights error:     {:.4e}".format(phase_weights_e))
    
    
    rho_err = np.mean(np.abs(np.sqrt(np.abs(amplitude_weights_true)**2) - np.sqrt(np.abs(amplitude_weights_pre)**2))) / np.mean(np.sqrt(np.abs(amplitude_weights_true)**2))
    print("mode amplitude error: {:.6e}".format(rho_err))
    
    
    #%% draw fields
    amplitude_dis_true = abs(np.squeeze(Image_complex.numpy()))
    phase_dis_true = np.angle(np.squeeze(Image_complex.numpy()))
    
    amplitude_dis_pre = abs(np.squeeze(pre_Images_rescale_comp))
    phase_dis_pre = np.angle(np.squeeze(pre_Images_rescale_comp))
    
    
    # amplitude_dis_true =np.real( np.squeeze(Image_complex.numpy()))
    # phase_dis_true = np.angle(np.squeeze(Image_complex.numpy()))
    
    # amplitude_dis_pre = np.real(np.squeeze(pre_Images_rescale_comp))
    # phase_dis_pre = np.angle(np.squeeze(pre_Images_rescale_comp))
    
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 8)
    
    ax0 = fig.add_subplot(gs[0, 0:2])  
    im0 = ax0.imshow(amplitude_dis_true, aspect='equal', cmap='viridis')
    ax0.axis('off')
    ax0.set_title('Ground Truth-Amplitude distribution')
    fig.colorbar(im0, ax=ax0)
    
    ax1 = fig.add_subplot(gs[0, 3:5])
    im1 = ax1.imshow(amplitude_dis_pre, aspect='equal', cmap='viridis')
    ax1.axis('off')
    ax1.set_title('Reconstruction')
    fig.colorbar(im1, ax=ax1)
    
    
    ax2 = fig.add_subplot(gs[0, 6:8])
    im2 = ax2.imshow(abs(amplitude_dis_true - amplitude_dis_pre), aspect='equal', cmap='viridis',vmin = 0,vmax = amplitude_dis_true.max())
    ax2.axis('off')
    ax2.set_title('Residual')
    fig.colorbar(im2, ax=ax2)
    
    
    ax3 = fig.add_subplot(gs[1, 0:2]) 
    im3 = ax3.imshow(phase_dis_true, aspect='equal', cmap='viridis')
    ax3.axis('off')
    ax3.set_title('Ground Truth-Phase distribution')
    fig.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(gs[1, 3:5]) 
    im4 = ax4.imshow(phase_dis_pre, aspect='equal', cmap='viridis')
    ax4.axis('off')
    ax4.set_title('Reconstruction')
    fig.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(gs[1, 6:8])  
    im5 = ax5.imshow(phase_dis_true - (phase_dis_pre), aspect='equal', cmap='viridis',vmin = -2*np.pi,vmax = 2*np.pi)
    ax5.axis('off')
    ax5.set_title('Residual')
    fig.colorbar(im5, ax=ax5)
    
    
    ax6 = fig.add_subplot(gs[2, 3:5])  
    im6 = ax6.imshow(-phase_dis_pre, aspect='equal', cmap='viridis')
    ax6.axis('off')
    ax6.set_title('Reconstruction (Conj)')
    fig.colorbar(im6, ax=ax6)
    
    
    ax7 = fig.add_subplot(gs[2, 6:8])  
    im7 = ax7.imshow(phase_dis_true - (-phase_dis_pre), aspect='equal', cmap='viridis',vmin = -2*np.pi,vmax = 2*np.pi)
    ax7.axis('off')
    ax7.set_title('Residual (Conj)')
    fig.colorbar(im7, ax=ax7)
    
    modes_plot = np.arange(1, num_modes + 1)
    
    
    # Amplitudes weights
    ax8 = fig.add_subplot(gs[3, 0:2])
    ax8.plot(modes_plot,amplitude_weights_true, 'o', label='True value')
    ax8.plot(modes_plot, amplitude_weights_pre, 'x', label='Prediction')
    ax8.vlines(modes_plot, amplitude_weights_true, amplitude_weights_pre, colors='gray', linestyles='dashed', label='Amplitude error')
    ax8.set_xlabel('Mode number')
    ax8.set_ylabel('Amplitude')
    ax8.set_title('Mode Amplitudes')
    ax8.set_ylim([0, 1])
    ax8.set_xlim([0, num_modes+1])
    
    ax8.legend()
    ax8.grid(True)
    
    # Phase weights
    ax9 = fig.add_subplot(gs[3, 3:5])
    ax9.plot(modes_plot,phase_weights_true, 'o', label='cos(True phase)')
    ax9.plot(modes_plot,phase_weights_pre, 'x', label='cos(Recovered phase)')
    ax9.vlines(modes_plot,phase_weights_true,phase_weights_pre, colors='gray', linestyles='dashed', label='cos(Phase) error')
    ax9.set_xlabel('Mode number')
    ax9.set_ylabel('Phase')
    ax9.set_title('Mode Phases')
    ax9.set_ylim([-np.pi, np.pi])
    ax9.set_xlim([0, num_modes+1])
    # ax9.legend()
    ax9.grid(True)
    
       
    # cos values of phases
    ax10 = fig.add_subplot(gs[3, 6:8])
    ax10.plot(modes_plot,np.cos( phase_weights_true), 'o', label='cos(True phase)')
    ax10.plot(modes_plot,np.cos( phase_weights_pre), 'x', label='cos(Recovered phase)')
    ax10.vlines(modes_plot,np.cos( phase_weights_true),np.cos( phase_weights_pre), colors='gray', linestyles='dashed', label='cos(Phase) error')
    ax10.set_xlabel('Mode number')
    ax10.set_ylabel('cos(Phase)')
    ax10.set_title('Mode Phases')
    ax10.set_ylim([-1.1, 1.1])
    ax10.set_xlim([0, num_modes+1])
    # ax10.legend()
    ax10.grid(True)
    
        
    plt.savefig('./results/result_' +str(num_modes)+'_modes.png', dpi=600)  # saves as fig.png
    
    plt.show()


    
    

