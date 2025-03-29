    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import numpy as np
from recon_image import recon_prediction
from recon_image import recon_prediction_tensor
import mmf_build_image
import matplotlib.pyplot as plt
#%% training function
def train(model: nn.Module,cuda_index,flag_all_cpus,num_epochs,
          train_loader,
          optimizer,criterion,flag_validation,flag_test,test_fr,phase_variants,mode_fields,
          Loss_epoch,Loss_iteration,Loss_val,Loss_val_index,training_cc,
          lr_step,lr_ratio,training_start, threshold ):
    
    # training on GPUs
    # device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    device = torch.device(cuda_index)
    # print(device)
    model.to(device)
    
    if flag_all_cpus == 1:
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs.")
            model = nn.DataParallel(model)

    #     net = nn.DataParallel(model, device_ids=[3,4])
 

    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_ratio)
    number_of_modes = mode_fields.shape[0]
    image_size = mode_fields.shape[1]
   
    
    print('Start training:')
    # test_images, test_labels = next(iter(test_loader))
    for epoch in range(num_epochs):
        running_loss = 0.0
        val_loss = 0.0
        for i, (images, labels) in enumerate(train_loader,0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # images = images.reshape(1,1,32,32)
            model.output_fc.register_forward_hook(get_activation)
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            # print("outputs:",outputs)
            
            # loss = criterion(abs(outputs), images)
            loss = criterion(abs(outputs), images[0,0,:,:])
            # loss = Variable(loss, requires_grad = True)
        
            # loss = criterion(outputs, labels)
            
            # print(loss)
            # save the training process --> visualization through matlab 
            Loss_iteration.append(loss.item())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            # save the loss once per epoch 
            running_loss += loss.item()     # extract the loss value
            # if epoch>1000:                
            #     plt.suptitle('Multi_Image') # 图片名称
            #     plt.subplot(1,3,1), plt.title('image')
            #     plt.imshow(images[0,0,:,:].cpu().detach().numpy())
            #     plt.subplot(1,3,2), plt.title('output')
            #     # plt.imshow(abs(outputs[0,0,:,:].cpu().detach().numpy()))  
            #     plt.imshow(abs(outputs.cpu().detach().numpy())) 
            if i % len(train_loader) == (len(train_loader)-1):    
                # print every 1000 (twice per epoch)    
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / len(train_loader) ))
                Loss_epoch.append(running_loss / len(train_loader) )
                # zero the loss            
                running_loss = 0.0  
                # plt.suptitle('Multi_Image') # 图片名称
                # plt.subplot(1,3,1), plt.title('image')
                # plt.imshow(images[0,0,:,:].cpu().detach().numpy())
                # plt.subplot(1,3,2), plt.title('output')
                # # plt.imshow(abs(outputs[0,0,:,:].cpu().detach().numpy()))  
                # plt.imshow(abs(outputs.cpu().detach().numpy()))          
        if loss.item() < threshold:
            print('Loss is below the threshold, stop training.')
            plt.suptitle('Multi_Image') # 图片名称
            plt.subplot(1,3,1), plt.title('image')
            plt.imshow(images[0,0,:,:].cpu().detach().numpy())
            plt.subplot(1,3,2), plt.title('output')
            # plt.imshow(abs(outputs[0,0,:,:].cpu().detach().numpy()))  
            plt.imshow(abs(outputs.cpu().detach().numpy())) 
            break
        
        if epoch % test_fr == 1 or epoch == 0 or test_fr == 1:  # validation each 5 epochs
        #     if flag_validation == 1:     

        #         # outputs = model(test_images.to(device))
        #         # val_loss = criterion(outputs,test_labels.to(device))
        #         # print('validation loss: %.4f' % val_loss)
        #         # Loss_val.append(val_loss.item())
        #         val_network(model,val_loader,device,criterion,Loss_val)            
        #         Loss_val_index.append(epoch+1)
            if flag_test == 1:              
                cc_N = test_network(model,train_loader,device,phase_variants,mode_fields)  
                training_cc.append(cc_N)
                # print(cc_N[0:10])
                # print(corr_N[0][0:10])
        if epoch > 997:
            plt.suptitle('Multi_Image') # 图片名称
            plt.subplot(1,3,1), plt.title('image')
            plt.imshow(images[0,0,:,:].cpu().detach().numpy())
            plt.subplot(1,3,2), plt.title('output')
            # plt.imshow(abs(outputs[0,0,:,:].cpu().detach().numpy()))  
            plt.imshow(abs(outputs.cpu().detach().numpy())) 
            
        scheduler.step()
        time_elapsed = time.time() - training_start
        print('Training lasted {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # return corr_N
    return activation

#%% validation
def val_network(model: nn.Module,val_loader,device,criterion,Loss_val):
    
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
def test_network( model:nn.Module, data_loader ,device,phase_variants,mode_fields):

    test_images, test_labels = next(iter(data_loader))
    num_modes = (test_labels.size(1) +1 )/2
    try:
        pred_vectors = model(test_images.to(device))
        # print("done 1")
    except:    
        pred_vectors = np.full_like(test_labels,0) 
        pred_vectors = []
        for index_test, (test_images,test_labels) in enumerate(data_loader,0):
    

            pred_vector = model(test_images.to(device))
            pred_vectors.append(pred_vector)
        
        print("done 2")
    # ssss = time.time()
    
    # corr_N, weights_complex = recon_prediction_tensor(
    #     num_modes,device,test_labels.size(0),pred_vectors,phase_variants,test_images,mode_fields)
    
    # eeee = time.time() - ssss
    # print("tensor result:",eeee)
    
    # ssss2 = time.time()
    
    corr_N, weights_complex = recon_prediction(
        num_modes,test_labels.size(0),pred_vectors.cpu().detach().numpy(),phase_variants,test_images.detach().numpy(),mode_fields)
    # eeee2 = time.time() - ssss2
    # print("normal result", eeee2)
    
    return corr_N
#     pred = MTNet_prediction(num_modes,test,32,trained_model_path)l

# #%% reconstruction 
# mode_fields_path = main_folder_path + "mode_fields_"+str(num_modes)+"modes.mat"
# mode_fields_mat = mat73.loadmat(mode_fields_path)
# mode_fields = mode_fields_mat["modes_field"]        #complex array

# corr_N, weights_complex = recon_prediction(num_modes,num_test_data,pred,phase_variants,image_test,mode_fields)
# plt.figure(99)
# plt.plot(corr_N)

# complex_fields = recon_vectors(weights_complex,mode_fields)

def get_activation(layer, input, output):
    global activation
    activation = output.cpu().detach().numpy()


    
