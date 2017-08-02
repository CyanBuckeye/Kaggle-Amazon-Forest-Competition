#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:37:07 2017

@author: andrew

Function:
load pretrained resnet model and finetune it on Amazon Forest dataset.
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
import random
from scipy.misc import imresize

#mean=[0.485, 0.456, 0.406]
#std=[0.229, 0.224, 0.225]

'''
normalization parameters of Resent trained on ImageNet dataset
I applied it to preprocessing of Amazon Forest dataset. However, it does harm to the result 
'''

net = torchvision.models.resnet50(pretrained=True)  

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=17):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.add_module('conv1', net.conv1)
        self.add_module('bn1', net.bn1)
        self.add_module('relu', net.relu)
        self.add_module('maxpool', net.maxpool)
        self.add_module('layer1', net.layer1)
        self.add_module('layer2', net.layer2)
        self.add_module('layer3', net.layer3)
        self.add_module('layer4_0', net.layer4[0])
        self.add_module('layer4_1', net.layer4[1])
        self.add_module('layer4_2', net.layer4[2])
        #self.add_module('avgpool', net.avgpool)#load pretrained pooling layer
        
        self.avgpool = nn.AvgPool2d(7)#train pooling layer from scratch
        
        self.fc = nn.Linear(2048, 17)#one fc layer
        
        '''
        self.fc =  nn.Sequential(OrderedDict([
                ('drop', nn.Dropout(0.2)),
                ('fc1', nn.Linear(2048, 512)),
                ('relu_fc1', nn.PReLU(512)), 
                ('fc2', nn.Linear(512, 17))
                ]))
        '''#two fc layers

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
    
def resnet34():
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model 

def resnet50():
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

mynet = resnet50()   
mynet = mynet.cuda()


no_gradList = [mynet.conv1.parameters(), mynet.bn1.parameters(), mynet.relu.parameters()
, mynet.maxpool.parameters(), mynet.layer1.parameters(), mynet.layer2.parameters(), mynet.layer3.parameters(),mynet.layer4_0.parameters(),mynet.layer4_1.parameters(),mynet.layer4_2.parameters()]  

for parameters in no_gradList:
    for param in parameters:
      param.requires_grad = False #freeze conv layers

data_path = ''#path of training data
valData_path = ''#path of validation data
channel = 3 #channel of image
imgSize = 256#size of image
inputSize = 256#size of data fed to CNN. In PyTorch, data in the form of N*C*H*W, where N is the batch number, C is the channel, H is height and W is width.n Here, H = W = inputSize
category = ['cloudy', 'partly_cloudy', 'haze', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground', 'primary', 'slash_burn'\
            , 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'clear']#class

label_path = 'labelMatrix.p'#mapping from Image name to its label
f = open(label_path, 'rb')
label_map = pickle.load(f)
f.close()

trainh5 = h5py.File(data_path, 'r')
data = trainh5['trainData'][:]
label = trainh5['trainLabel'][:]
trainh5.close()

valh5 = h5py.File('valData_path' ,'r')
val_data = valh5['valData'][:]
val_label = valh5['valLabel'][:]
valh5.close()

show = 30
epoch = 20
miniBatch = 32 #I tried different batch sizes, such as 64,96 and 128. Among them, 32 performs best
testNum = 8100 # I choose 8100 images for validataion

trainNum = 32379 # number of images for training
batch = trainNum / miniBatch
testbatch = testNum / miniBatch
loss_accum = 0.0
val_interval = 1

loss_array = [] #list storing the training loss
print('finish data loading')
resumeModel = ''#path of model you want to resume training
load_flag = True

if load_flag == True:
    mynet.load_state_dict(torch.load(resumeModel))
order = np.zeros((trainNum,), dtype=np.int32) #numpy array helping to shuffle data

y_axis = [] # list storing the validation loss
base_lr = 0.1 # learning rate of last several layers
tune_lr = 0.01# learning rate of conv layers

for i in range(trainNum):
    order[i] = i
temp_momentum = 0.9
for e in range(epoch):
    if e % val_interval == 0 and e > 0:
        mynet.eval()#evaluation mode 
        score = 0.0
        for i in range(testNum):
            img = val_data[i].reshape((1, channel, imgSize, imgSize))
            temp_data = img
            temp_data = temp_data.astype(np.float32)
            temp_data = temp_data / float(255)
            #temp_data[:,0] = (temp_data[:,0] - mean[0]) / std[0]
            #temp_data[:,1] = (temp_data[:,1] - mean[1]) / std[1]
            #temp_data[:,2] = (temp_data[:,2] - mean[2]) / std[2]

            temp_data = torch.from_numpy(temp_data)
            test_data = Variable(temp_data).cuda()
            
            temp_label = val_label[i]
            temp_label = temp_label.astype(np.float32)
            temp_label = torch.from_numpy(temp_label)
            getlabel = Variable(temp_label).cuda()
            output = mynet(test_data)
            output = F.sigmoid(output)
            loss = F.binary_cross_entropy(output, getlabel)
            score += loss.cpu().data.numpy()[0]
            
        print('the test score of epoch %d is %.4f' % (e, score / float(testNum)))
        y_axis.append(score / float(testNum))
        
        
        torch.save(mynet.state_dict(), './Kaggle_forest' + '_%.4f' % (score / float(testNum)) + '.pth')#store the intermediate model
    optimizer = optim.SGD([
        {'params': mynet.conv1.parameters(), 'lr': tune_lr},
        {'params': mynet.bn1.parameters(), 'lr':  tune_lr},
        {'params': mynet.relu.parameters(), 'lr': tune_lr},
        {'params': mynet.maxpool.parameters(), 'lr': tune_lr},
        {'params': mynet.layer1.parameters(), 'lr': tune_lr},
        {'params': mynet.layer2.parameters(), 'lr': tune_lr},    
        {'params': mynet.layer3.parameters(), 'lr':   tune_lr},
        {'params': mynet.layer4_0.parameters(), 'lr': tune_lr},
        {'params': mynet.layer4_1.parameters(), 'lr': tune_lr}, 
        {'params': mynet.layer4_2.parameters(), 'lr': tune_lr}, 
        
        {'params': mynet.avgpool.parameters()},   
        {'params': mynet.fc.parameters()}                 
    ], lr=base_lr, momentum=temp_momentum)
    
    np.random.shuffle(order)
    temp_Traindata = np.zeros((miniBatch, channel, inputSize, inputSize), dtype=np.uint8)
    temp_Trainlabel = np.zeros((miniBatch, len(category)), dtype=np.uint8)
    
    mynet.train()
    loss_accum = 0.0
    for i in range(batch):
        for j in range(miniBatch):
            img = data[order[i * miniBatch + j]]
            rd2 = np.random.randint(4)
            
            if rd2 == 0:#rotate image randomly
               pass
            if rd2 == 1:
               img = np.rot90(img, 1)
            if rd2 == 2:
                img = np.rot90(img, 2)
            if rd2 == 3:
                img = np.rot90(img, 3)
                
            rd4 = np.random.randint(3) #resize and randomly crop 
            if rd4 == 0:
                img = imresize(img, 1.03)
                x = random.randint(0, img.shape[0] - inputSize)
                y = random.randint(0, img.shape[0] - inputSize)
                img = img[x : x + inputSize, y : y + inputSize]
            img = img.transpose((2,0,1))
            img = img.astype(np.float32)
            img = img.reshape((1, channel, inputSize, inputSize))
            rd1 = np.random.randint(4)
            
            if rd1 == 0:#randomly flip
                temp_Traindata[j] = img
            if rd1 == 1:
                temp_Traindata[j] = img[:, :, ::-1, :]
            if rd1 == 2:
                temp_Traindata[j] = img[:, :, :, ::-1]
            if rd1 == 3:
                temp_Traindata[j] = img[:, :, ::-1, ::-1]
                
            temp_Trainlabel[j] = label[order[i * miniBatch + j]]
  
        
        temp_data = temp_Traindata
        temp_data = temp_data.astype(np.float32)
        temp_data = temp_data / float(255)
        #temp_data[:,0] = (temp_data[:,0] - mean[0]) / std[0]
        #temp_data[:,1] = (temp_data[:,1] - mean[1]) / std[1]
        #temp_data[:,2] = (temp_data[:,2] - mean[2]) / std[2]
        temp_data = torch.from_numpy(temp_data)
        train_data = Variable(temp_data).cuda()
        temp_label = temp_Trainlabel
        temp_label = temp_label.astype(np.float32)
        temp_label = torch. from_numpy(temp_label)
        train_label = Variable(temp_label).cuda()
        
        optimizer.zero_grad()
        outputs = mynet(train_data)
        outputs = outputs.view(outputs.size(0), outputs.size(1))
        outputs = F.sigmoid(outputs)

        loss = F.binary_cross_entropy(outputs, train_label)
        loss.backward()
        optimizer.step()
        loss_accum += loss.data[0]
        if (i + 1) % show == 0:
           print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(e+1, epoch, i+1, batch, loss_accum / show))
           loss_array.append(loss_accum / show)
           loss_accum = 0.0
    
       
           
overall_array = np.zeros((len(category),), dtype=np.int32)
correct_array = np.zeros((len(category),), dtype=np.int32)
judge_array = np.zeros((len(category), ), dtype=np.int32)
precision_array = np.zeros((len(category),), dtype=np.float32)
recall_array = np.zeros((len(category),), dtype=np.float32)
sum_array = np.zeros((len(category),),dtype=np.int32)
conf_array = np.zeros((testNum, 8, len(category)))
score = 0.0
mynet.eval()
for i in range(testNum):
    img = val_data[i]
    
    test_data = np.zeros((8, channel, imgSize, imgSize), dtype=np.uint8)
    temp_data = img.transpose((1,2,0))#test with TTA(test time augmentation): test on augmented validation images and vote final label
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
    
   
    temp_label = val_label[i]
    output = mynet(test_data)
    output = F.sigmoid(output)
    output = output.cpu().data.numpy()[...]
    conf_array[i] = output
   
    
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if len(output[:,j][output[:,j] >= 0.15]) >= 5:
            predict[j] = 1
    tp = 0
    fp = 0
    tf = 0
    ff = 0
    for j in range(len(category)):
        if predict[j] == 1:
            judge_array[j] += 1
        if temp_label[j] == 1:
            overall_array[j] += 1
        if predict[j] == 1 and temp_label[j] == 1:
            tp += 1
            correct_array[j] += 1
        if predict[j] == 0 and temp_label[j] == 0:
            tf += 1
        if predict[j] == 1 and temp_label[j] == 0:
            fp += 1
        if predict[j] == 0 and temp_label[j] == 1:
            ff += 1
    if tp == 0:
        continue
    else:
        p = tp / float(tp + fp)
        r = tp / float(tp + ff)
        score += 5 * p * r / (4 * p + r)
for i in range(len(category)):
    recall_array[i] = correct_array[i] / float(overall_array[i])#get recall rate and precision rate for every class
    precision_array[i] = correct_array[i] / float(judge_array[i])
print('finish')
