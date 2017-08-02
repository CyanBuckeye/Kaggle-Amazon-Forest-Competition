#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:24:03 2017

@author: xu.2727
test with UNet
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

class UNet(nn.Module):
    def __init__(self, inputSize):
        super(UNet, self).__init__()
        channel, width, height = inputSize
           
        self.block1 = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(4, 24, kernel_size=3, stride=1)),
                ('BN1_1', nn.BatchNorm2d(24)),  
                ('relu1_1', nn.PReLU(24)),
                ('conv1_4', nn.Conv2d(24, 24, kernel_size=3, stride=1)),
                ('BN1_4', nn.BatchNorm2d(24)),
                ('relu1_4', nn.PReLU(24))
                ]))    
                     
        self.block2 = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(24, 48, kernel_size=2, stride=1)),
                ('BN2_1', nn.BatchNorm2d(48)),
                ('relu2_1', nn.PReLU(48)),
                ('conv2_4', nn.Conv2d(48, 48, kernel_size=2, stride=1)),
                ('BN2_4', nn.BatchNorm2d(48)),
                ('relu2_4', nn.PReLU(48))
                ]))  
        self.block3 = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(48, 96, kernel_size=2, stride=1)),
                ('BN3_1', nn.BatchNorm2d(96)),
                ('relu3_1', nn.PReLU(96)),
                ('conv3_4', nn.Conv2d(96, 96, kernel_size=2, stride=1)),
                ('BN3_4', nn.BatchNorm2d(96)),
                ('relu3_4', nn.PReLU(96))
                ]))  
    
        self.block4 = nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(96, 192, kernel_size=2, stride=1)),
                ('BN4_1', nn.BatchNorm2d(192)),
                ('relu4_1', nn.PReLU(192)),
                ('conv4_4', nn.Conv2d(192, 192, kernel_size=2, stride=1)),
                ('BN4_4', nn.BatchNorm2d(192)),
                ('relu4_4', nn.PReLU(192))
                ]))        
        self.block5 = nn.Sequential(OrderedDict([
                ('conv5_1', nn.Conv2d(192, 384, kernel_size=2, stride=1)),
                ('BN5_1', nn.BatchNorm2d(384)),
                ('relu5_1', nn.PReLU(384)),
                ('conv4_4', nn.Conv2d(384, 384, kernel_size=2, stride=1)),
                ('BN4_4', nn.BatchNorm2d(384)),
                ('relu4_4', nn.PReLU(384))
                ]))      
        self.block6 = nn.Sequential(OrderedDict([
                ('dropout6_1', nn.Dropout2d(p=0.3)),
                ('conv6_1', nn.Conv2d(in_channels=576, out_channels=192, kernel_size=2, stride=1)),
                ('relu6_1', nn.PReLU(192)),
                ('conv6_2', nn.Conv2d(in_channels=192, out_channels=192, kernel_size=2, stride=1)),
                ('relu6_2', nn.PReLU(192))
        ]))
        self.block7 = nn.Sequential(OrderedDict([
                ('dropout7_1', nn.Dropout2d(p=0.3)),
                ('conv7_1', nn.Conv2d(in_channels=288, out_channels=96, kernel_size=2, stride=1)),
                ('relu7_1', nn.PReLU(96)),
                ('conv7_2', nn.Conv2d(in_channels=96, out_channels=96, kernel_size=2, stride=1)),
                ('relu7_2', nn.PReLU(96))
        ]))      
        self.block8 = nn.Sequential(OrderedDict([
                ('dropout8_1', nn.Dropout2d(p=0.3)),
                ('conv8_1', nn.Conv2d(in_channels=144, out_channels=48, kernel_size=2, stride=1)),
                ('relu8_1', nn.PReLU(48)),
                ('conv8_2', nn.Conv2d(in_channels=48, out_channels=48, kernel_size=2, stride=1)),
                ('relu8_2', nn.PReLU(48))
        ]))      
        self.block9 = nn.Sequential(OrderedDict([
                ('dropout9_1', nn.Dropout2d(p=0.3)),
                ('conv9_1', nn.Conv2d(in_channels=72, out_channels=24, kernel_size=2, stride=1)),
                ('relu9_1', nn.PReLU(24)),
                ('conv9_2', nn.Conv2d(in_channels=24, out_channels=24, kernel_size=2, stride=1)),
                ('relu9_2', nn.PReLU(24)) 
        ]))
        self.block10 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(744, 1024)),
        ('relu_fc1', nn.PReLU(1024)), 
        ('fc_dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(1024, 512)),
        ('relu_fc2', nn.PReLU(512)),
        ('fc_dropout2', nn.Dropout(p=0.5)),
        ('score', nn.Linear(512, len(category)))
        ]))

    def forward(self, x):
        blk1 = self.block1(x)

        pool1 = F.max_pool2d(blk1, kernel_size=2, stride=2)
        
        blk2 = self.block2(pool1)
        pool2 = F.max_pool2d(blk2, kernel_size=2, stride=2)
        
        blk3 = self.block3(pool2)
        pool3 = F.max_pool2d(blk3, kernel_size=2, stride=2)
        
        blk4 = self.block4(pool3)
        pool4 = F.max_pool2d(blk4, kernel_size=2, stride=2)
        
            
        blk5 = self.block5(pool4)
        pool5 = F.upsample_bilinear(blk5, scale_factor=2)
        gl1 = F.max_pool2d(blk5, kernel_size=blk5.size()[2:])
        
        
        cat1 = torch.cat((blk4[:,:,2:26,2:26], pool5), dim=1)
        
        blk6 = self.block6(cat1)
        pool6 = F.upsample_bilinear(blk6, scale_factor=2)
        gl2 = F.max_pool2d(blk6, kernel_size=blk6.size()[2:])
        
        cat2 = torch.cat((pool6, blk3[:,:,8:52,8:52]), dim=1)
        blk7 = self.block7(cat2)
        pool7 = F.upsample_bilinear(blk7, scale_factor=2)
        gl3= F.max_pool2d(blk7, kernel_size=blk7.size()[2:])
          
        cat3 = torch.cat((pool7, blk2[:,:,20:104,20:104]), dim=1)
        blk8 = self.block8(cat3)
        pool8 = F.upsample_bilinear(blk8, scale_factor=2)
        gl4 = F.max_pool2d(blk8, kernel_size=blk8.size()[2:])
        
        cat4 = torch.cat((pool8, blk1[:,:,44:208,44:208]), dim=1)
        blk9= self.block9(cat4)
        gl5 = F.max_pool2d(blk9, kernel_size=blk9.size()[2:])
        
        feature = torch.cat((gl1, gl2, gl3, gl4, gl5), dim=1)
        feature = feature.view(feature.size(0), -1)
      
        logit = self.block10(feature)
        logit = logit.view(logit.size(0), logit.size(1))
        prob = F.sigmoid(logit)
        return prob

Unet = UNet([channel,imgSize,imgSize])
Unet.cuda()
modelPath = ''
Unet.load_state_dict(torch.load(modelPath))
Unet.eval()

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
   
    output = Unet(test_data)  
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
    test_data = test_data / float(255)
    test_data = torch.from_numpy(test_data)
    test_data = Variable(test_data).cuda()
    output = Unet(test_data)
   
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