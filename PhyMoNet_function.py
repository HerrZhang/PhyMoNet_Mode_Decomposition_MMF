#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:14:04 2023

@author: qzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import numpy as np
from recon_image import recon_prediction
from recon_image import recon_prediction_tensor
import mmf_functions
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CorrCoef import corr2,corr2_tensor
import normalization
import math


#%% training functionnew 
def train_PhyMoNet(model,optimizer,criterion,cuda_index, 
                   Image,Label,loss_threshold,
                   num_epochs,lr_step, lr_ratio, 
                   flag_test, test_fr,mode_fields):
    # input of DNN
    # output of DNN 
    # label: weights_vector C x 2N-1 
    # prediction 2N-1: amplitude + phase-1
    num_data = len(Image)
    
    batch_size = 1
    #data to DataLoader
    train_dataset = torch.utils.data.TensorDataset(Image, Label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(cuda_index)
    # print(device)
    model.to(device)
    
    if cuda_index == 'cuda:999':
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs.")
            model = nn.DataParallel(model)
            
    # for i, (images, labels) in enumerate(train_loader,0):
    for index_data, (Input_Image, Output_Label) in enumerate(train_loader,0):
                
        Loss_epoch, training_cc_index,training_cc = [[] for ii in range(3)]
       
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_ratio)        
        training_start = time.time()
        print('Start training:')

        # training based on one sample
        for epoch in range(num_epochs):

            # get the inputs
            images = Variable(Input_Image.to(device))
            labels = Variable(Output_Label.to(device))
            
            # images = images.reshape(1,1,32,32)
            model.output_fc.register_forward_hook(get_activation)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(images)            
            
            # loss = criterion(abs(outputs), images)
            loss = criterion(abs(outputs), images[0,0,:,:])
            # print(loss)
            # save the training process --> visualization through matlab 
            Loss_epoch.append(loss.item())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            if epoch % 100 == 0:    
                # print every 1000 (twice per epoch)    
                print('[data: %5d, epoch: %5d] loss: %.6f' %(index_data + 1,epoch + 1,  loss))
                # Loss_epoch.append(loss)

            if loss.item() < loss_threshold:
                print('Loss is below the loss_threshold, stop training.')
                plt.suptitle('Multi_Image') # 图片名称
                plt.subplot(1,3,1), plt.title('image')
                plt.imshow(images[0,0,:,:].cpu().detach().numpy())
                plt.subplot(1,3,2), plt.title('output')
                # plt.imshow(abs(outputs[0,0,:,:].cpu().detach().numpy()))  
                plt.imshow(abs(outputs.cpu().detach().numpy())) 
                break

            # training_start3 = time.time()
            if epoch % test_fr == 1 or epoch == 0 or test_fr == 1:  # validation each 5 epochs
                # print('test here 1')
                if flag_test == 1:   
                    cc_N = test_PhyMoNet(model,Input_Image,Output_Label,device,mode_fields)                      

                    training_cc.append(cc_N)
               
                    training_cc_index.append(epoch+1) # save the index of the CC. For matlab, epoch starts from 1

            # adjust learning rate
            scheduler.step()
     
    return activation,outputs,training_cc_index,training_cc,Loss_epoch   

#%% validation
def val_PhyMoNet(model: nn.Module,val_loader,device,criterion,Loss_val):
    
    running_loss_val = 0.0
    for index_val, (val_images, val_labels) in enumerate(val_loader,0):
        # get the inputs
        val_images = Variable(val_images.to(device))
        val_labels = Variable(val_labels.to(device))
        
        val_outputs = model(val_images)
        loss_val = criterion(val_outputs,val_labels.to(device))
        running_loss_val += loss_val.item()     # extract the loss value
    
    loss_val_mean = running_loss_val / len(val_loader)        
    Loss_val.append(loss_val_mean)
    return print('validation loss: %.4f' % loss_val_mean)
    
    # Loss_val_index.append(epoch+1)
        

#%% test network
def test_PhyMoNet( model:nn.Module, test_images, test_labels,device,mode_fields):
# =============================================================================
#     test single input 
# =============================================================================

    # test_images, test_labels = next(iter(data_loader))
    # test_images = data_loader
    num_modes = mode_fields.shape[0]
    try:

        pred_field= model(test_images.to(device))
        # print("done 1")

    except:    
        # pred_field = np.full_like(test_labels,0) 
        pred_field = []
        # for index_test, (test_images,test_labels) in enumerate(data_loader,0):
    
            # pred_vector = model(test_images.to(device))
            # pred_field.append(pred_vector)

        pred_vector = model(test_images.to(device))
        pred_field.append(pred_vector)
        
        print("done 2")
    
    corr_N = corr2_tensor(test_images[0,0,:,:], abs(pred_field))
    # print('mean corr: {:.4f}%'.format(corr_N.mean()*100))

    return corr_N

#%% test network transfer learning 

#%% built the light field from prediction (vectors )
def recon_prediction_PhyMoNet(num_modes, num_test_data, pred, image_test, mode_fields):
    
    num_modes = int(num_modes)
  

    weights_complex = np.zeros((num_test_data, num_modes), dtype='complex')
    corr_N = np.zeros([num_test_data, 1])

    # time_begin = time.time()
    for i in range(num_test_data):
     
        ground_truth_i = image_test[i, 0, :, :]
      
        corr_N[i] = corr2(ground_truth_i, abs(pred))
        weights_complex[i, :] = 1

    print('mean corr: {:.4f}%'.format(corr_N.mean()*100))

    # time_elapsed = time.time() - time_begin
    # print('recon_prediction_PhyMoNet Elapsed {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    
    return corr_N, weights_complex




#%% test network
def test_PhyMoNet_dataloader( model:nn.Module, data_loader,device,phase_variants,mode_fields):
# =============================================================================
#     test single input 
# =============================================================================

    test_images, test_labels = next(iter(data_loader))
    # test_images = data_loader
    num_modes = mode_fields.shape[0]
    try:
        pred_field = model(test_images.to(device))
        # print("done 1")
    except:    
        pred_field = np.full_like(test_labels,0) 
        pred_field = []

        pred_vector = model(test_images.to(device))
        pred_field.append(pred_vector)
        
        print("done 2")
    
    corr_N, weights_complex = recon_prediction(
        num_modes,test_labels.size(0),pred_field.cpu().detach().numpy(),phase_variants,test_images.detach().numpy(),mode_fields)

    # def mmf_build_image(num_mode, image_size, num_data, complex_weights):

        

    return corr_N

#%% rescale predicted vector
def rescal_result_PhyMoNet(pred_vector,num_modes,image_size):
    
    # amp = pred_vector[0,0:num_modes].reshape(1, num_modes)
    amp = pred_vector[0,0:num_modes]
    phase = pred_vector[0,num_modes:]
    amp_n = amp*amp
    # amp_n = normalization.normalization(amp, 0, 1)

    amp_n = amp_n/ np.linalg.norm(amp_n) 
    # amp_n[amp_n < 0] = 0
    for index, item in enumerate(phase):
        if item >= 0:
            phase[index]  = phase[index]  % (2*math.pi)
        if item < 0 :
            while phase[index]  < 0 :
                phase[index]   = phase[index]+(2*math.pi)
    
    rel_phs = np.zeros((1,num_modes))
    
    # rel_phs = np.concatenate(([0], phase), axis=0)
    for mode_index0 in range(0,num_modes-1):
        # rel_phs[0,mode_index0+1] = phase_n[mode_index0]
        rel_phs[0,mode_index0+1] = phase[mode_index0]
    
    # phase = math.asin(math.sin(phase))
    # pred_vector_rescale = np.concatenate((amp_n, rel_phs), axis=0)
    pred_vector_rescale = np.concatenate((amp_n, phase), axis=0)
    
    complex_weights = amp_n*np.exp(1j*rel_phs) 
    Image,Image_complex = mmf_functions.mmf_build_image(num_modes, image_size, 1, complex_weights)

    
    return pred_vector_rescale,Image,Image_complex,complex_weights

#%% calculate amp_error and pha_error and put result together

#only for debug
# pre_vectors = pred_vector_rescale

# label_vectors = Label

def analyses_results(num_modes,pre_vectors,label_vectors):
    amp_error = []
    phase_error = []
    amp_vector= label_vectors.numpy()[0,0:num_modes]
    phase_vector = label_vectors.numpy()[0,num_modes:]
    amp_pred= pre_vectors[0:num_modes]
    phase_pred = pre_vectors[num_modes:]
    
    amp_error = abs(amp_vector-amp_pred)
    
    phase_error1 = abs(phase_vector-phase_pred)
    phase_error2 = abs((2*math.pi-phase_vector)-phase_pred)
    
    phase_error = min([np.mean(phase_error1),np.mean(phase_error2)]) / (2*math.pi)
    if np.mean(phase_error1)<=np.mean(phase_error2):
        phase_error = phase_error1/(2*math.pi)
    else:
        phase_error = phase_error2/(2*math.pi)
    
    phase_error_corr_N = np.zeros_like(phase_pred)
    phase_pred_corr = np.zeros_like(phase_pred)
    
    for index_i in range(len(phase_pred)):

        phase_error_corr_N[index_i] = min(phase_error1[index_i],phase_error2[index_i])/(2*math.pi)
        
        if phase_error1[index_i] <= phase_error2[index_i]:
            phase_pred_corr[index_i] = phase_pred[index_i]            
        else:
            phase_pred_corr[index_i] = 2*math.pi-phase_pred[index_i]
    return amp_error,phase_error,phase_error_corr_N

#%% get the predicted vector
def get_activation(layer, input, output):
    global activation
    activation = output.cpu().detach().numpy()