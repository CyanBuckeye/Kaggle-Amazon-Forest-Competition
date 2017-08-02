#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:53:21 2017

@author: xu.2727
"""

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import numpy as np
import h5py
import os
from skimage.external.tifffile import imread
from skimage.external.tifffile import imsave
from PIL import Image
import matplotlib.pyplot as plt

channel = 4
imgSize = 256
category = ['cloudy', 'partly_cloudy', 'haze', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground', 'primary', 'slash_burn'\
            , 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'clear']
class Net(nn.Module):
    def __init__(self, inputSize):
        super(Net, self).__init__()
        channel, width, height = inputSize 
        self.block1 = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(4, 24, kernel_size=3, stride=1, groups=4)),
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
                ('conv4_1', nn.Conv2d(96, 256, kernel_size=2, stride=1, dilation=2)),
                ('BN4_1', nn.BatchNorm2d(256)),
                ('relu4_1', nn.PReLU(256)),
                ('conv4_4', nn.Conv2d(256, 256, kernel_size=2, stride=1, dilation=2)),
                ('BN4_4', nn.BatchNorm2d(256)),
                ('relu4_4', nn.PReLU(256))
                ]))        
        self.block5 = nn.Sequential(OrderedDict([
                ('conv5_1', nn.Conv2d(256, 512, kernel_size=2, stride=1, dilation=3)),
                ('BN5_1', nn.BatchNorm2d(512)),
                ('relu5_1', nn.PReLU(512)),
                ('conv4_4', nn.Conv2d(512, 512, kernel_size=2, stride=1, dilation=3)),
                ('BN4_4', nn.BatchNorm2d(512)),
                ('relu4_4', nn.PReLU(512))
                ]))      
       
        self.block6 = nn.Sequential(OrderedDict([
                ('dropout6_1', nn.Dropout2d(p=0.3)),
                ('conv6_1', nn.Conv2d(in_channels=768, out_channels=256, kernel_size=2, stride=1)),
                ('relu6_1', nn.PReLU(256)),
                ('conv6_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)),
                ('relu6_2', nn.PReLU(256))
        ]))
        self.block7 = nn.Sequential(OrderedDict([
                ('dropout7_1', nn.Dropout2d(p=0.3)),
                ('conv7_1', nn.Conv2d(in_channels=352, out_channels=96, kernel_size=2, stride=1)),
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
        ('fc1', nn.Linear(936, 512)),
        ('relu_fc1', nn.PReLU(512)), 
        ('fc_dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(512, 512)),
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
        
        
        cat1 = torch.cat((blk4[:,:,6:20,6:20], pool5), dim=1)
        
        blk6 = self.block6(cat1)
        pool6 = F.upsample_bilinear(blk6, scale_factor=2)
        gl2 = F.max_pool2d(blk6, kernel_size=blk6.size()[2:])
        
        cat2 = torch.cat((pool6, blk3[:,:,18:42,18:42]), dim=1)
        blk7 = self.block7(cat2)
        pool7 = F.upsample_bilinear(blk7, scale_factor=2)
        gl3= F.max_pool2d(blk7, kernel_size=blk7.size()[2:])
        
        cat3 = torch.cat((pool7, blk2[:,:,30:74,30:74]), dim=1)
        blk8 = self.block8(cat3)
        pool8 = F.upsample_bilinear(blk8, scale_factor=2)
        gl4 = F.max_pool2d(blk8, kernel_size=blk8.size()[2:])
        
        cat4 = torch.cat((pool8, blk1[:,:,84:168,84:168]), dim=1)
        blk9= self.block9(cat4)
        gl5 = F.max_pool2d(blk9, kernel_size=blk9.size()[2:])
        
        feature = torch.cat((gl1, gl2, gl3, gl4, gl5), dim=1)
        feature = feature.view(feature.size(0), -1)
      
        logit = self.block10(feature)
        logit = logit.view(logit.size(0), logit.size(1))
        prob = F.sigmoid(logit)
        return prob

print('start')   
fig, ax = plt.subplots()
net = Net([channel,imgSize,imgSize])
net.cuda()

base_lr = 0.5
print("begin load")

train_path = ''
val_path = ''
label_path = 'labelMatrix.p'
f = open(label_path, 'rb')
label_map = pickle.load(f)
f.close()


trainh5 = h5py.File(train_path, 'r')
data = trainh5['trainData'][:]
label = trainh5['trainLabel'][:]
trainh5.close()

valh5 = h5py.File(val_path,'r')
val_data = valh5['valData'][:]
val_label = valh5['valLabel'][:]
valh5.close()

show = 30
epoch = 120
miniBatch = 32 #32 96 128
testNum = 8100

trainNum = 32379
batch = trainNum / miniBatch
testbatch = testNum / miniBatch
loss_accum = 0.0


loss_array = []
print('finish data loading')
model_path = ''
load_flag = True
if load_flag == True:
    net.load_state_dict(torch.load(model_path))
order = np.zeros((trainNum,), dtype=np.int32)

for i in range(trainNum):
    order[i] = i
         
for e in range(epoch):
    optimizer = torch.optim.SGD(net.parameters(), lr = base_lr)
    if e == 35:
       base_lr = 0.1
    if e == 50:
        base_lr = 0.05
    if e == 75:
        base_lr = 0.01
    if e == 100:
        base_lr = 0.001
    if e % 10 == 0 and e > 0:
        net.eval()
        score = 0.0
        for i in range(testNum):
            img = val_data[i].reshape((1, channel, imgSize, imgSize))
            temp_data = img
            temp_data = temp_data.astype(np.float32)
            temp_data = torch.from_numpy(temp_data)
            test_data = Variable(temp_data).cuda()
            
            temp_label = val_label[i]
            temp_label = temp_label.astype(np.float32)
            temp_label = torch. from_numpy(temp_label)
            getlabel = Variable(temp_label).cuda()
            output = net(test_data)
            loss = F.binary_cross_entropy(output, getlabel)
            score += loss.cpu().data.numpy()[0]
            
        print('the test score of epoch %d is %.4f' % (e, score / float(testNum)))
        y_axis.append(score / float(testNum))
        
        torch.save(net.state_dict(), './deconv_NVDI_dilation/dilation_Kaggle_256' + '_%.4f' % (score / float(testNum)) + '.pth')
    
    np.random.shuffle(order)
    paths = os.listdir(train_path)
    temp_Traindata = np.zeros((miniBatch, channel, imgSize, imgSize), dtype=np.uint8)
    temp_Trainlabel = np.zeros((miniBatch, len(category)), dtype=np.uint8)
    
    net.train()
    loss_accum = 0.0
    for i in range(batch):
        for j in range(miniBatch):
            img = data[order[i * miniBatch + j]]
            rd2 = np.random.randint(4)
          
            if rd2 == 0:
               pass
            if rd2 == 1:
               img = np.rot90(img, 1)
            if rd2 == 2:
                img = np.rot90(img, 2)
            if rd2 == 3:
                img = np.rot90(img, 3)
            img = img.transpose((2,0,1))
            img = img.astype(np.float32)
            img = img.reshape((1, channel, imgSize, imgSize))
            rd1 = np.random.randint(4)
            #data augmentation
            
            if rd1 == 0:
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
        temp_data = torch.from_numpy(temp_data)
        train_data = Variable(temp_data).cuda()
        temp_label = temp_Trainlabel
        temp_label = temp_label.astype(np.float32)
        temp_label = torch. from_numpy(temp_label)
        train_label = Variable(temp_label).cuda()
        
        
        optimizer.zero_grad()
        outputs = net(train_data)
        loss = F.binary_cross_entropy(outputs, train_label)
        loss.backward()
        optimizer.step()
        loss_accum += loss.data[0]
        if (i + 1) % show == 0:
           print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(e+1, epoch, i+1, batch, loss_accum / show))
           loss_array.append(loss_accum / show)
           loss_accum = 0.0
    
        
       
            
ax.plot(x_axis, y_axis)
plt.show()            
#last test

overall_array = np.zeros((len(category),), dtype=np.int32)
correct_array = np.zeros((len(category),), dtype=np.int32)
judge_array = np.zeros((len(category), ), dtype=np.int32)
precision_array = np.zeros((len(category),), dtype=np.float32)
recall_array = np.zeros((len(category),), dtype=np.float32)
sum_array = np.zeros((len(category),),dtype=np.int32)
conf_array = []
for i in range(len(category)):
    conf_array.append([])
score = 0.0
net.eval()
for i in range(testNum):
    img = val_data[i]
    img = img.astype(np.float32)
    img = img.reshape((1,channel,imgSize,imgSize))
    temp_data = np.zeros((1, channel, imgSize, imgSize))
    temp_data = img
    temp_data = temp_data.astype(np.float32)
    temp_data = torch.from_numpy(temp_data)
    test_data = Variable(temp_data).cuda()
    
   
    temp_label = val_label[i]
    output = net(test_data)
    
    output = output.cpu().data.numpy()[0]
    for i in range(len(category)):
        conf_array[i].append(output[i])
        if temp_label[i] == 1:
            sum_array[i] += 1
    
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if output[j] >= 0.1:
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
    recall_array[i] = correct_array[i] / float(overall_array[i])
    precision_array[i] = correct_array[i] / float(judge_array[i])
print('finish')

for i in range(len(category)):
    conf_array[i] = np.asarray(conf_array[i])
    
