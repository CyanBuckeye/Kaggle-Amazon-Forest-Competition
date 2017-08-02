#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:24:03 2017

@author: xu.2727
test with deconvNet
input: test images (in the form of HDF5 dataset), well-trained network
output: prediction labels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import pickle
import torch.nn.functional as F
import h5py
import math
from collections import OrderedDict
import csv

csv_file = open(os.path.join(csv_path), 'w') #output CSV file
fieldnames = ['image_name', 'tags']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

writer.writeheader()

channel = 4
imgSize = 256
category = ['cloudy', 'partly_cloudy', 'haze', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground', 'primary', 'slash_burn'\
            , 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'clear']
output_category = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 
            'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
correspond_l = [6, 11, 10, 16, 9, 0, 13, 8, 2, 12, 15, 14, 3, 7, 1, 4, 5]

class deconvNVDI_Net(nn.Module):
    def __init__(self, inputSize):
        super(deconvNVDI_Net, self).__init__()
        channel, width, height = inputSize
     
        self.block1_nvdi =  nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(1, 4, kernel_size=11, stride=1)),
                ('BN1_1', nn.BatchNorm2d(4)),
                ('relu1_1', nn.PReLU(4)),
                ('conv1_2', nn.Conv2d(4, 4, kernel_size=1, stride=1)),
                ('BN1_2', nn.BatchNorm2d(4)),
                ('relu1_2', nn.PReLU(4)),
                ('conv1_3', nn.Conv2d(4, 4, kernel_size=1, stride=1)),
                ('BN1_3', nn.BatchNorm2d(4)),
                ('relu1_3', nn.PReLU(4)),
                ]))     
                               
        self.block1 = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(3, 32, kernel_size=11, stride=1)),
                ('BN1_1', nn.BatchNorm2d(32)),
                ('relu1_1', nn.PReLU(32)),
                ('conv1_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_2', nn.BatchNorm2d(32)),
                ('relu1_2', nn.PReLU(32)),
                ('conv1_3', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_3', nn.BatchNorm2d(32)),
                ('relu1_3', nn.PReLU(32)),
                ]))    
                     
        self.block2_nvdi = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(4, 8, kernel_size=4, stride=1)),
                ('BN2_1', nn.BatchNorm2d(8)),
                ('relu2_1', nn.PReLU(8)),
                ('conv2_2', nn.Conv2d(8, 8, kernel_size=1, stride=1)),
                ('BN2_2', nn.BatchNorm2d(8)),
                ('relu2_2', nn.PReLU(8)),
                ('conv2_3', nn.Conv2d(8, 8, kernel_size=1, stride=1)),
                ('BN2_3', nn.BatchNorm2d(8)),
                ('relu2_3', nn.PReLU(8)),
                ]))
        
        self.block2 = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
                ('BN2_1', nn.BatchNorm2d(64)),
                ('relu2_1', nn.PReLU(64)),
                ('conv2_2', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_2', nn.BatchNorm2d(64)),
                ('relu2_2', nn.PReLU(64)),
                ('conv2_3', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_3', nn.BatchNorm2d(64)),
                ('relu2_3', nn.PReLU(64)),
                ]))  
        
        self.block3_nvdi = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(8, 16, kernel_size=3, stride=1)),
                ('BN3_1', nn.BatchNorm2d(16)),
                ('relu3_1', nn.PReLU(16)),
                ('conv3_2', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN3_2', nn.BatchNorm2d(16)),
                ('relu3_2', nn.PReLU(16)),
                ('conv3_3', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN3_3', nn.BatchNorm2d(16)),
                ('relu3_3', nn.PReLU(16)),
                ('conv3_4', nn.Conv2d(16, 16, kernel_size=3, stride=1)),
                ('BN3_4', nn.BatchNorm2d(16)),
                ('relu3_4', nn.PReLU(16))
        ]))
        self.block3 = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
                ('BN3_1', nn.BatchNorm2d(128)),
                ('relu3_1', nn.PReLU(128)),
                ('conv3_2', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN3_2', nn.BatchNorm2d(128)),
                ('relu3_2', nn.PReLU(128)),
                ('conv3_3', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN3_3', nn.BatchNorm2d(128)),
                ('relu3_3', nn.PReLU(128)),
                ('conv3_4', nn.Conv2d(128, 128, kernel_size=3, stride=1)),
                ('BN3_4', nn.BatchNorm2d(128)),
                ('relu3_4', nn.PReLU(128))
                ]))  

        self.block4_nvdi = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(16, 32, kernel_size=3, stride=1)),
                ('BN3_1', nn.BatchNorm2d(32)),
                ('relu3_1', nn.PReLU(32)),
                ('conv3_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN3_2', nn.BatchNorm2d(32)),
                ('relu3_2', nn.PReLU(32)),
                ('conv3_3', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN3_3', nn.BatchNorm2d(32)),
                ('relu3_3', nn.PReLU(32)),
                ('conv3_4', nn.Conv2d(32, 32, kernel_size=3, stride=1)),
                ('BN3_4', nn.BatchNorm2d(32)),
                ('relu3_4', nn.PReLU(32))
        ]))
        
        self.block4 = nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(128, 256, kernel_size=3, stride=1)),
                ('BN4_1', nn.BatchNorm2d(256)),
                ('relu4_1', nn.PReLU(256)),
                ('conv4_2', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_2', nn.BatchNorm2d(256)),
                ('relu4_2', nn.PReLU(256)),
                ('conv4_3', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_3', nn.BatchNorm2d(256)),
                ('relu4_3', nn.PReLU(256)),
                ('conv4_4', nn.Conv2d(256, 256, kernel_size=3, stride=1)),
                ('BN4_4', nn.BatchNorm2d(256)),
                ('relu4_4', nn.PReLU(256))
                ]))          
        
        self.block5 = nn.Sequential(OrderedDict([
                ('deconv5_1', nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0)),
                ('BN5_1', nn.BatchNorm2d(128)),
                ('relu5_1', nn.PReLU(128)),
                ('upSample5_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))
        self.block6 = nn.Sequential(OrderedDict([
                ('deconv6_1', nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=5, stride=1, padding=0, output_padding=0)),
                ('BN6_1', nn.BatchNorm2d(64)),
                ('relu6_1', nn.PReLU(64)),
                ('upSample6_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))
        self.block7 = nn.Sequential(OrderedDict([
                ('deconv7_1', nn.ConvTranspose2d(in_channels=192, out_channels=32, kernel_size=5, stride=1, padding=0, output_padding=0)),
                ('BN7_1', nn.BatchNorm2d(32)),
                ('relu7_1', nn.PReLU(32)),
                ('upSample7_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))      
        self.block8 = nn.Sequential(OrderedDict([
                ('deconv8_1', nn.ConvTranspose2d(in_channels=96, out_channels=16, kernel_size=9, stride=1, padding=0, output_padding=0)),
                ('BN8_1', nn.BatchNorm2d(16)),
                ('relu8_1', nn.PReLU(16)),
                ('upSample8_1', nn.UpsamplingBilinear2d(scale_factor=2)),
        ]))      
        self.block9 = nn.Sequential(OrderedDict([
                ('conv9_1', nn.Conv2d(16, 32, kernel_size=1, stride=1)),
                ('BN9_1', nn.BatchNorm2d(32)),
                ('relu9_1', nn.PReLU(32)),
                ('conv9_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN9_2', nn.BatchNorm2d(32)),
                ('relu9_2', nn.PReLU(32)),
                ('score', nn.Conv2d(32, 5, kernel_size=1, stride=1))
                ]))
        
        self.block10 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(528 * 1 * 1, 512)),#nvdi 528; otherwise 496
        ('relu_fc1', nn.PReLU(512)), 
        ('fc2', nn.Linear(512, 512)),
        ('relu_fc2', nn.PReLU(512)), 
        ('score', nn.Linear(512, len(category)))
        ]))

    def forward(self, x):
        blk1 = self.block1(x[:,:3,:,:])
        blk1_nvdi = self.block1_nvdi(x[:,3,:,:].resize(x.size()[0],1,x.size()[2], x.size()[3]))
        
        
        pool1 = F.max_pool2d(blk1, kernel_size=2, stride=2)
        pool1_nvdi = F.max_pool2d(blk1_nvdi, kernel_size=2, stride=2)
        
        blk2 = self.block2(pool1)
        pool2 = F.max_pool2d(blk2, kernel_size=2, stride=2)
        gl1 = F.max_pool2d(pool2, kernel_size=pool2.size()[2:])
        
        blk_nvdi2 = self.block2_nvdi(pool1_nvdi)
        pool2_nvdi = F.max_pool2d(blk_nvdi2, kernel_size=2, stride=2)
        
        blk3 = self.block3(pool2)
        pool3 = F.max_pool2d(blk3, kernel_size=2, stride=2)
        gl2 = F.max_pool2d(pool3, kernel_size=pool3.size()[2:])
        
        blk_nvdi3 = self.block3_nvdi(pool2_nvdi)
        pool3_nvdi = F.max_pool2d(blk_nvdi3, kernel_size=2, stride=2)
        
        blk4 = self.block4(pool3)
        pool4 = F.max_pool2d(blk4, kernel_size=2, stride=2)
        
        blk_nvdi4 = self.block4_nvdi(pool3_nvdi)
        gl_nvdi = F.max_pool2d(blk_nvdi4, kernel_size=blk_nvdi4.size()[2:])

        gl1 = F.max_pool2d(pool4, kernel_size=pool4.size()[2:])
        
        blk5 = self.block5(pool4)
        gl2 = F.max_pool2d(blk5, kernel_size=blk5.size()[2:])
        
       
        cat1 = torch.cat((blk4, blk5), dim=1)
        blk6 = self.block6(cat1)
        gl3 = F.max_pool2d(blk6, kernel_size=blk6.size()[2:])
        
        cat2 = torch.cat((blk6, blk3), dim=1)
        blk7 = self.block7(cat2)
        gl4 = F.max_pool2d(blk7, kernel_size=blk7.size()[2:])
        
        cat3 = torch.cat((blk7, blk2), dim=1)
        blk8 = self.block8(cat3)
        gl5 = F.max_pool2d(blk8, kernel_size=blk8.size()[2:])
        
       
        feature = torch.cat((gl1, gl2, gl3, gl4, gl5, gl_nvdi), dim=1)
        feature = feature.view(feature.size(0), -1)
      
        logit = self.block10(feature)
        logit = logit.view(logit.size(0), logit.size(1))
        prob = F.sigmoid(logit)
        return prob
    
fig, ax = plt.subplots()
x_axis = np.arange(50)
y_axis = []
deconvNVDI_net = deconvNVDI_Net([channel,imgSize,imgSize])
deconvNVDI_net.cuda()

modelPath=''
deconvNVDI_net.load_state_dict(torch.load(modelPath))
deconvNVDI_net.eval()

testData_path = ''
h5_test = h5py.File(testData_path, 'r')
testData= h5_test['Data'][...]
h5_test.close()

additionData_path = ''
h5_testAddition = h5py.File(additionData_path, 'r')
AddData = h5_testAddition['Data'][...]
h5_testAddition.close()

testNum = testData.shape[0]
testAddNum = AddData.shape[0]

for i in range(testNum):
    print(i)
    temp_data = testData[i]
    temp_data = temp_data.transpose((1,2,0))
    test_data = np.zeros((8, channel, imgSize, imgSize), dtype=np.uint8)
    
    test_data[0] = temp_data.transpose((2,0,1))
    test_data[1] = test_data[0][:,::-1,::-1]
    test_data[2] = np.rot90(temp_data, 1).transpose((2,0,1))
    test_data[3] = test_data[2][:,::-1,::-1]
    test_data[4] = np.rot90(temp_data, 2).transpose((2,0,1))
    test_data[5] = test_data[4][:,::-1,::-1]
    test_data[6] = np.rot90(temp_data, 3).transpose((2,0,1))
    test_data[7] = test_data[5][:,::-1,::-1]
    
    test_data = test_data.astype(np.float32)
    test_data = torch.from_numpy(test_data)
    test_data = Variable(test_data).cuda()
   
    output = deconvNVDI_net(test_data)
    output = output.cpu().data.numpy()[...]
    total = 0
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if len(output[:,j][output[:,j] >= 0.15]) >= 5:
         predict[correspond_l[j]] = 1
         total += 1
         
    write_category = ""
    for j in range(len(predict)):
        if predict[j] == 1:
            write_category += output_category[j]
            total -= 1
            if total > 0:
                write_category += ' '
    writer.writerow({'image_name': 'test_' + str(i), 'tags': write_category})
    
for i in range(testAddNum):
    print(i + testNum)
    temp_data = AddData[i]
    temp_data = temp_data.transpose((1,2,0))
    test_data = np.zeros((8, channel, imgSize, imgSize), dtype=np.uint8)
    
    test_data[0] = temp_data.transpose((2,0,1))
    test_data[1] = test_data[0][:,::-1,::-1]
    test_data[2] = np.rot90(temp_data, 1).transpose((2,0,1))
    test_data[3] = test_data[2][:,::-1,::-1]
    test_data[4] = np.rot90(temp_data, 2).transpose((2,0,1))
    test_data[5] = test_data[4][:,::-1,::-1]
    test_data[6] = np.rot90(temp_data, 3).transpose((2,0,1))
    test_data[7] = test_data[5][:,::-1,::-1]
    
    test_data = test_data.astype(np.float32)
    test_data = torch.from_numpy(test_data)
    test_data = Variable(test_data).cuda()
    
    output = deconvNVDI_net(test_data)
    output = output.cpu().data.numpy()[...]
    
    total = 0
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if len(output[:,j][output[:,j] >= 0.15]) >= 5:
         predict[correspond_l[j]] = 1
         total += 1
    write_category = ""
    for j in range(len(predict)):
        if predict[j] == 1:
            write_category += output_category[j]
            total -= 1
            if total > 0:
                write_category += ' '
    writer.writerow({'image_name': 'file_' + str(i), 'tags': write_category})
csv_file.close()

print('finish')