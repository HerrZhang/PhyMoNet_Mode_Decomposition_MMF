# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:54:13 2022

@author: sy
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time

class MLP_MD(nn.Module):
    def __init__(self, num_modes,input_size,mode_fields):
        super().__init__()

        self.input_fc = nn.Linear(input_size*input_size, 1024)
        # self.hidden_fc = nn.Linear(1024, 1024)
        self.hidden_fc = nn.Linear(1024, 256)
        self.output_fc = nn.Linear(256, num_modes*2-1)
        # self.act_fun1 = nn.Softmax()
        # self.act_fun2 = nn.Sigmoid()
        # self.act_fun3 = nn.Tanh()
        # self.act_fun4 = nn.ReLU()
        self.params1 = nn.ParameterDict({
               'num_modes': num_modes,
               'image_size':input_size,
               'mode_fields':mode_fields     
       })

    def forward(self, x):
                

        num_modes = self.params1['num_modes']
        image_size = self.params1['image_size']
        mode_fields = self.params1['mode_fields']    

        
        x = x.view(x.shape[0], -1)
        
                


        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        # x = F.relu(self.output_fc(x))


        # training_start4 = time.time() 
        x = self.output_fc(x)
        # x = self.act_fun1(x)
        # x[0,0:num_modes] = self.act_fun1(x[0,0:num_modes])
        # x[0,num_modes] = self.act_fun2(x[0,num_modes])
        # x[0,num_modes+1:] = self.act_fun2(x[0,num_modes+1:])
        # print('x_value:',x.cpu().detach().numpy())
        # x = F.hardtanh(x,0,1)
        
        
        # time_elapsed = time.time() - training_start4
        # print('reconstruct time2:',time_elapsed)

        
        device = x.device
        
        # x = MLP_MD.cal_image(x, num_modes, image_size, mode_fields, device)
        x = MLP_MD.cal_image_new(x, num_modes, image_size, mode_fields, device)

        return x
    
    def cal_image(x,num_modes,image_size,mode_fields,device):

        mode_fields = torch.from_numpy(mode_fields).to(device)
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)

        for mode_index0 in range(0,num_modes-1):
            new_phase[mode_index0+1] = phase[mode_index0]
        # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
        complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
        for mode_index in range(0, num_modes):
            # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 

            single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data

        x = single_Image_data
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
      
    def cal_image_new(x,num_modes,image_size,mode_fields,device):
        # mmf_modes = np.zeros((image_size, image_size, num_modes), dtype=complex)
        # for z in range(0, num_modes):
        #     mmf_modes[:, :, z] = mode_fields[z,:,:]
            
        mmf_modes = torch.from_numpy(mode_fields).to(device)
        
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)

        for mode_index0 in range(0,num_modes-1):
            new_phase[mode_index0+1] = phase[mode_index0]
        # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
        complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
        single_Image_data= complex_vector*mmf_modes
        # for mode_index in range(0, num_modes):
        #     # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 

        #     single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data

        x = single_Image_data.sum(2)
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
    # def cal_image(x,num_modes,image_size,mode_fields,device):

    #     mode_fields = torch.from_numpy(mode_fields).to(device)
    #     amp = x[0,0:num_modes]
    #     amp_n =amp /torch.norm(amp)
    #     phase = x[0,num_modes:]
    #     phase_n = phase /torch.norm(phase)*2*torch.pi
    #     single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
            
    #     new_phase = torch.zeros(num_modes)

    #     for mode_index0 in range(0,num_modes-1):
    #         new_phase[mode_index0+1] = phase_n[mode_index0]
    #         complex_vector = amp_n*(torch.exp(1j*new_phase).to(device))
    #     for mode_index in range(0, num_modes):
    #         single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
    #     x = single_Image_data
    #         #images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
       
    #     return x
    def cal_image_new_tl(x,num_modes,image_size,mode_fields,device):
        # mmf_modes = np.zeros((image_size, image_size, num_modes), dtype=complex)
        # for z in range(0, num_modes):
        #     mmf_modes[:, :, z] = mode_fields[z,:,:]
            
        mmf_modes = torch.from_numpy(mode_fields).to(device)
        num_data = x.size(1)
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        pre_Image_data = torch.zeros((num_data,1,image_size, image_size)).to(device)    
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)
        for index_data in range(num_data):
            
            for mode_index0 in range(0,num_modes-1):
                new_phase[mode_index0+1] = phase[mode_index0]
            # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
            complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
            single_Image_data= complex_vector*mmf_modes
            # for mode_index in range(0, num_modes):
            #     # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
            #     single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data
        
            pre_Image_data[index_data,:,:,:] = single_Image_data.sum(2)
            x =     pre_Image_data
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
    # def cal_image(x,num_modes,image_size,mode_fields,device):

    #     mode_fields = torch.from_numpy(mode_fields).to(device)
    #     amp = x[0,0:num_modes]
    #     amp_n =amp /torch.norm(amp)
    #     phase = x[0,num_modes:]
    #     phase_n = phase /torch.norm(phase)*2*torch.pi
    #     single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
            
    #     new_phase = torch.zeros(num_modes)

    #     for mode_index0 in range(0,num_modes-1):
    #         new_phase[mode_index0+1] = phase_n[mode_index0]
    #         complex_vector = amp_n*(torch.exp(1j*new_phase).to(device))
    #     for mode_index in range(0, num_modes):
    #         single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
    #     x = single_Image_data
    #         #images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
       
    #     return x
    


class MLP_MD_tl(nn.Module):
    def __init__(self, num_modes,input_size,mode_fields):
        super().__init__()

        self.input_fc = nn.Linear(input_size*input_size, 1024)
        # self.hidden_fc = nn.Linear(1024, 1024)
        self.hidden_fc = nn.Linear(1024, 256)
        self.output_fc = nn.Linear(256, num_modes*2-1)
        # self.act_fun1 = nn.Softmax()
        # self.act_fun2 = nn.Sigmoid()
        # self.act_fun3 = nn.Tanh()
        # self.act_fun4 = nn.ReLU()
        self.params1 = nn.ParameterDict({
               'num_modes': num_modes,
               'image_size':input_size,
               'mode_fields':mode_fields     
       })

    def forward(self, x):
                

        num_modes = self.params1['num_modes']
        image_size = self.params1['image_size']
        mode_fields = self.params1['mode_fields']    

        
        x = x.view(x.shape[0], -1)
        
                


        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        # x = F.relu(self.output_fc(x))


        # training_start4 = time.time() 
        x = self.output_fc(x)
        # x = self.act_fun1(x)
        # x[0,0:num_modes] = self.act_fun1(x[0,0:num_modes])
        # x[0,num_modes] = self.act_fun2(x[0,num_modes])
        # x[0,num_modes+1:] = self.act_fun2(x[0,num_modes+1:])
        # print('x_value:',x.cpu().detach().numpy())
        # x = F.hardtanh(x,0,1)
        
        
        # time_elapsed = time.time() - training_start4
        # print('reconstruct time2:',time_elapsed)

        
        device = x.device
        
        # x = MLP_MD.cal_image(x, num_modes, image_size, mode_fields, device)
        x = MLP_MD.cal_image_new_tl(x, num_modes, image_size, mode_fields, device)

        return x
    
    def cal_image(x,num_modes,image_size,mode_fields,device):

        mode_fields = torch.from_numpy(mode_fields).to(device)
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)

        for mode_index0 in range(0,num_modes-1):
            new_phase[mode_index0+1] = phase[mode_index0]
        # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
        complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
        for mode_index in range(0, num_modes):
            # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 

            single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data

        x = single_Image_data
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
      
    def cal_image_new(x,num_modes,image_size,mode_fields,device):
        # mmf_modes = np.zeros((image_size, image_size, num_modes), dtype=complex)
        # for z in range(0, num_modes):
        #     mmf_modes[:, :, z] = mode_fields[z,:,:]
            
        mmf_modes = torch.from_numpy(mode_fields).to(device)
        
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)

        for mode_index0 in range(0,num_modes-1):
            new_phase[mode_index0+1] = phase[mode_index0]
        # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
        complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
        single_Image_data= complex_vector*mmf_modes
        # for mode_index in range(0, num_modes):
        #     # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 

        #     single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data

        x = single_Image_data.sum(2)
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
    # def cal_image(x,num_modes,image_size,mode_fields,device):

    #     mode_fields = torch.from_numpy(mode_fields).to(device)
    #     amp = x[0,0:num_modes]
    #     amp_n =amp /torch.norm(amp)
    #     phase = x[0,num_modes:]
    #     phase_n = phase /torch.norm(phase)*2*torch.pi
    #     single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
            
    #     new_phase = torch.zeros(num_modes)

    #     for mode_index0 in range(0,num_modes-1):
    #         new_phase[mode_index0+1] = phase_n[mode_index0]
    #         complex_vector = amp_n*(torch.exp(1j*new_phase).to(device))
    #     for mode_index in range(0, num_modes):
    #         single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
    #     x = single_Image_data
    #         #images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
       
    #     return x
    def cal_image_new_tl(x,num_modes,image_size,mode_fields,device):
        # mmf_modes = np.zeros((image_size, image_size, num_modes), dtype=complex)
        # for z in range(0, num_modes):
        #     mmf_modes[:, :, z] = mode_fields[z,:,:]
            
        mmf_modes = torch.from_numpy(mode_fields).to(device)
        num_data = x.size(1)
        amp = torch.zeros((1,num_modes),dtype=torch.double)
        amp = x[0,0:num_modes]
        squ_amp = amp*amp
        phase = x[0,num_modes:]
        # single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
        pre_Image_data = torch.zeros((num_data,1,image_size, image_size)).to(device)    
        single_Image_data = torch.zeros((image_size, image_size)).to(device)    
        new_phase = torch.zeros(num_modes)
        for index_data in range(num_data):
            
            for mode_index0 in range(0,num_modes-1):
                new_phase[mode_index0+1] = phase[mode_index0]
            # complex_vector = amp*(torch.exp(1j*new_phase).to(device))
            complex_vector = squ_amp*(torch.exp(1j*new_phase).to(device))
            single_Image_data= complex_vector*mmf_modes
            # for mode_index in range(0, num_modes):
            #     # single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
            #     single_Image_data = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data
        
            pre_Image_data[index_data,:,:,:] = single_Image_data.sum(2)
            x =     pre_Image_data
            # images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
        return x
    
    # def cal_image(x,num_modes,image_size,mode_fields,device):

    #     mode_fields = torch.from_numpy(mode_fields).to(device)
    #     amp = x[0,0:num_modes]
    #     amp_n =amp /torch.norm(amp)
    #     phase = x[0,num_modes:]
    #     phase_n = phase /torch.norm(phase)*2*torch.pi
    #     single_Image_data = torch.zeros((1,1,image_size, image_size)).to(device)
            
    #     new_phase = torch.zeros(num_modes)

    #     for mode_index0 in range(0,num_modes-1):
    #         new_phase[mode_index0+1] = phase_n[mode_index0]
    #         complex_vector = amp_n*(torch.exp(1j*new_phase).to(device))
    #     for mode_index in range(0, num_modes):
    #         single_Image_data[0,0,:,:] = complex_vector[mode_index]*mode_fields[mode_index,:, :]+single_Image_data[0,0,:,:] 
        
    #     x = single_Image_data
    #         #images1 = images.cpu().detach().numpy(); from CorrCoef import corr2,corr2_tensor;corr2(images1,abs(single_Image_data))  
       
    #     return x