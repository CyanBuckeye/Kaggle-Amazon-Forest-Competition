#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:24:03 2017

@author: xu.2727
test with ResNet-50
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

channel = 3
imgSize = 256
category = ['cloudy', 'partly_cloudy', 'haze', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground', 'primary', 'slash_burn'\
            , 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'clear']
output_category = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 
            'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
correspond_l = [6, 11, 10, 16, 9, 0, 13, 8, 2, 12, 15, 14, 3, 7, 1, 4, 5]

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

loaded_net = torchvision.models.resnet50(pretrained=True) 

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=17):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.add_module('conv1', loaded_net.conv1)
        self.add_module('bn1', loaded_net.bn1)
        self.add_module('relu', loaded_net.relu)
        self.add_module('maxpool', loaded_net.maxpool)
        self.add_module('layer1', loaded_net.layer1)
        self.add_module('layer2', loaded_net.layer2)
        self.add_module('layer3', loaded_net.layer3)
        self.add_module('layer4_0', loaded_net.layer4[0])
        self.add_module('layer4_1', loaded_net.layer4[1])
        self.add_module('layer4_2', loaded_net.layer4[2])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, 17)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50():
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

net = resnet50()   
net = net.cuda()
modelPath = ''
net.load_state_dict(torch.load(modelPath))
net.eval()

testData_path = ''
h5_test = h5py.File(testData_path, 'r')
testData= h5_test['Data'][...][:,:3]
h5_test.close()

additionTest_path = ''
h5_testAddition = h5py.File(additionTest_path, 'r')
AddData = h5_testAddition['Data'][...][:,:3]
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
    test_data = test_data / float(255)
    test_data = torch.from_numpy(test_data)
    test_data = Variable(test_data).cuda()
    output = net(test_data)
    output = F.sigmoid(output)
    output = output.cpu().data.numpy()[...]
 
    total = 0
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if len(output[:,j][output[:,j] >= 0.15) >= 5:
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
    output = net(test_data)
    output = F.sigmoid(output)
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