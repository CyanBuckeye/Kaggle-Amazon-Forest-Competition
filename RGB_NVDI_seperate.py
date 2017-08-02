#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:37:21 2017

@author: xu.2727

Train deconvolution network from scratch
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
import matplotlib.pyplot as plt

channel = 4 #input channels are R,G,B and NDVI
imgSize = 72
category = ['cloudy', 'partly_cloudy', 'haze', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground', 'primary', 'slash_burn'\
            , 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down', 'clear']
class Net(nn.Module):
    def __init__(self, inputSize):
        super(Net, self).__init__()
        channel, width, height = inputSize
        #pre-process block
        self.preprocess = nn.Sequential(OrderedDict([
                ('conv0_1', nn.Conv2d(3, 16, kernel_size=1, stride=1)),
                ('BN0_1',   nn.BatchNorm2d(16)),
                ('relu0_1', nn.PReLU(16)),
                ('conv0_2', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN0_2', nn.BatchNorm2d(16)),
                ('relu0_2', nn.PReLU(16)),
                ('conv0_3', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN0_3', nn.BatchNorm2d(16)),
                ('relu0_3', nn.PReLU(16))
                ]
        ))
        
        self.preprocess_nvdi = nn.Sequential(OrderedDict([
                ('conv0_1', nn.Conv2d(1, 4, kernel_size=1, stride=1)),
                ('BN0_1',   nn.BatchNorm2d(4)),
                ('relu0_1', nn.PReLU(4)),
                ('conv0_2', nn.Conv2d(4, 4, kernel_size=1, stride=1)),
                ('BN0_2', nn.BatchNorm2d(4)),
                ('relu0_2', nn.PReLU(4)),
                ('conv0_3', nn.Conv2d(4, 4, kernel_size=1, stride=1)),
                ('BN0_3', nn.BatchNorm2d(4)),
                ('relu0_3', nn.PReLU(4))
                ]
        ))
                                 
        self.block1 = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(16, 32, kernel_size=11, stride=1)),
                ('BN1_1', nn.BatchNorm2d(32)),
                ('relu1_1', nn.PReLU(32)),
                ('conv1_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_2', nn.BatchNorm2d(32)),
                ('relu1_2', nn.PReLU(32)),
                ('conv1_3', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_3', nn.BatchNorm2d(32)),
                ('relu1_3', nn.PReLU(32)),
                ('conv1_4', nn.Conv2d(32, 32, kernel_size=9, stride=1)),
                ('BN1_4', nn.BatchNorm2d(32)),
                ('relu1_4', nn.PReLU(32))
                ])) 

        self.block1_nvdi = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(4, 8, kernel_size=11, stride=1)),
                ('BN1_1', nn.BatchNorm2d(8)),
                ('relu1_1', nn.PReLU(8)),
                ('conv1_2', nn.Conv2d(8, 8, kernel_size=1, stride=1)),
                ('BN1_2', nn.BatchNorm2d(8)),
                ('relu1_2', nn.PReLU(8)),
                ('conv1_3', nn.Conv2d(8, 8, kernel_size=1, stride=1)),
                ('BN1_3', nn.BatchNorm2d(8)),
                ('relu1_3', nn.PReLU(8)),
                ('conv1_4', nn.Conv2d(8, 8, kernel_size=9, stride=1)),
                ('BN1_4', nn.BatchNorm2d(8)),
                ('relu1_4', nn.PReLU(8))
                ]))                        

        self.block2 = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(40, 64, kernel_size=9, stride=1)),
                ('BN2_1', nn.BatchNorm2d(64)),
                ('relu2_1', nn.PReLU(64)),
                ('conv2_2', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_2', nn.BatchNorm2d(64)),
                ('relu2_2', nn.PReLU(64)),
                ('conv2_3', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_3', nn.BatchNorm2d(64)),
                ('relu2_3', nn.PReLU(64)),
                ('conv2_4', nn.Conv2d(64, 64, kernel_size=5, stride=1)),
                ('BN2_4', nn.BatchNorm2d(64)),
                ('relu2_4', nn.PReLU(64))
                ]))  
       
        self.block2_nvdi = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(8, 16, kernel_size=3, stride=1)),
                ('BN2_1', nn.BatchNorm2d(16)),
                ('relu2_1', nn.PReLU(16)),
                ('conv2_2', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN2_2', nn.BatchNorm2d(16)),
                ('relu2_2', nn.PReLU(16)),
                ('conv2_3', nn.Conv2d(16, 16, kernel_size=1, stride=1)),
                ('BN2_3', nn.BatchNorm2d(16)),
                ('relu2_3', nn.PReLU(16)),
                ('conv2_4', nn.Conv2d(16, 16, kernel_size=3, stride=1)),
                ('BN2_4', nn.BatchNorm2d(16)),
                ('relu2_4', nn.PReLU(16))
        ]))
        
        self.block3 = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(64, 72, kernel_size=3, stride=1)),
                ('BN3_1', nn.BatchNorm2d(72)),
                ('relu3_1', nn.PReLU(72)),
                ('conv3_2', nn.Conv2d(72, 72, kernel_size=1, stride=1)),
                ('BN3_2', nn.BatchNorm2d(72)),
                ('relu3_2', nn.PReLU(72)),
                ('conv3_3', nn.Conv2d(72, 72, kernel_size=1, stride=1)),
                ('BN3_3', nn.BatchNorm2d(72)),
                ('relu3_3', nn.PReLU(72)),
                ('conv3_4', nn.Conv2d(72, 72, kernel_size=3, stride=1)),
                ('BN3_4', nn.BatchNorm2d(72)),
                ('relu3_4', nn.PReLU(72))
                ]))  
        
        self.block3_nvdi = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(16, 32, kernel_size=5, stride=1)),
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
                ('conv4_1', nn.Conv2d(72, 128, kernel_size=3, stride=1)),
                ('BN4_1', nn.BatchNorm2d(128)),
                ('relu4_1', nn.PReLU(128)),
                ('conv4_2', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN4_2', nn.BatchNorm2d(128)),
                ('relu4_2', nn.PReLU(128)),
                ('conv4_3', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN4_3', nn.BatchNorm2d(128)),
                ('relu4_3', nn.PReLU(128)),
                ('conv4_4', nn.Conv2d(128, 128, kernel_size=3, stride=1)),
                ('BN4_4', nn.BatchNorm2d(128)),
                ('relu4_4', nn.PReLU(128))
                ]))    
        
        self.block4_2 = nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(128, 256, kernel_size=2, stride=1)),
                ('BN4_1', nn.BatchNorm2d(256)),
                ('relu4_1', nn.PReLU(256)),
                ('conv4_2', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_2', nn.BatchNorm2d(256)),
                ('relu4_2', nn.PReLU(256)),
                ('conv4_3', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_3', nn.BatchNorm2d(256)),
                ('relu4_3', nn.PReLU(256)),
                ('conv4_4', nn.Conv2d(256, 256, kernel_size=2, stride=1)),
                ('BN4_4', nn.BatchNorm2d(256)),
                ('relu4_4', nn.PReLU(256))
                ]))

        self.block4_nvdi = nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(32, 64, kernel_size=2, stride=1)),
                ('BN4_1', nn.BatchNorm2d(64)),
                ('relu4_1', nn.PReLU(64)),
                ('conv4_2', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN4_2', nn.BatchNorm2d(64)),
                ('relu4_2', nn.PReLU(64)),
                ('conv4_3', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN4_3', nn.BatchNorm2d(64)),
                ('relu4_3', nn.PReLU(64)),
                ('conv4_4', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN4_4', nn.BatchNorm2d(64)),
                ('relu4_4', nn.PReLU(64))
        ]))
        
        self.block5 = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(264 * 1 * 1, 512)),
                ('relu_fc1', nn.PReLU(512)), 
                ('fc2', nn.Linear(512, 512)),
                ('relu_fc2', nn.PReLU(512)), 
                ('score', nn.Linear(512, len(category)))
                ]))

    def forward(self, x):
        out = self.preprocess(x[:,:3,:,:])
        out_nvdi = self.preprocess_nvdi(x[:,3,:,:].resize(x.size()[0],1,x.size()[2], x.size()[3]))
        
        blk1 = self.block1(out)
        blk1_nvdi = self.block1_nvdi(out_nvdi)
        
        pool1 = F.max_pool2d(blk1, kernel_size=2, stride=2)
        pool1_nvdi = F.max_pool2d(blk1_nvdi, kernel_size=2, stride=2)
        
        f = torch.cat((pool1, pool1_nvdi),dim=1)
        blk2 = self.block2(f)
        pool2 = F.max_pool2d(blk2, kernel_size=2, stride=2)
        gl1 = F.max_pool2d(pool2, kernel_size=pool2.size()[2:])
        
        
        blk3 = self.block3(pool2)
        pool3 = F.max_pool2d(blk3, kernel_size=2, stride=2)
        
        gl2 = F.max_pool2d(pool3, kernel_size=pool3.size()[2:])
        blk4 = self.block4(pool3)
        pool4 = F.max_pool2d(blk4, kernel_size=2, stride=2)
        
        gl3 = F.max_pool2d(pool4, kernel_size=pool4.size()[2:])
       
        feature = torch.cat((gl1, gl2, gl3), dim=1)
        feature = feature.view(feature.size(0), -1)
        
        
        logit = self.block5(feature)
        logit = logit.view(logit.size(0), logit.size(1))
        prob = F.sigmoid(logit)
        return prob

fig, ax = plt.subplots()
x_axis = np.arange(50)
y_axis = []
net = Net([channel,imgSize,imgSize])
net.cuda()

base_lr = 0.5
print("begin load")

trainData_path = ''
trainh5 = h5py.File(trainData_path, 'r')
data = trainh5['trainData'][:]
label = trainh5['trainLabel'][:]
trainh5.close()

valData_path = ''
valh5 = h5py.File(valData_path ,'r')
val_data = valh5['valData'][:]
val_label = valh5['valLabel'][:]
valh5.close()

show = 20 # show training loss every 20 training batches
epoch = 50
miniBatch = 32 #batch size
num = data.shape[0]
batch = num / miniBatch
loss_accum = 0.0

testNum = val_data.shape[0]

print('finish data loading')
load_flag = False
if load_flag == True:
    net.load_state_dict(torch.load('./Kaggle_forest.pth'))
order = np.zeros((num,), dtype=np.int32)
for i in range(num):
    order[i] = i
for e in range(epoch):
    optimizer = torch.optim.SGD(net.parameters(), lr = base_lr)
    
    if epoch == 15:
        base_lr = base_lr / float(10)
    if epoch == 40:
        base_lr = base_lr / float(10)
    #test net
    
    net.eval()
    score = 0.0
    for i in range(testNum):
        temp_data = val_data[i]
        temp_data = temp_data.reshape((1, channel, imgSize, imgSize))
        temp_data = temp_data.astype(np.float32)
        temp_data = torch.from_numpy(temp_data)
        test_data = Variable(temp_data).cuda()
        
        temp_label = val_label[i]
        output = net(test_data)
        
        output = output.cpu().data.numpy()[0]
        predict = np.zeros((len(category),), dtype=np.uint8)
        for j in range(len(category)):
            if output[j] >= 0.1:
                predict[j] = 1
        tp = 0
        fp = 0
        tf = 0
        ff = 0
        for j in range(len(category)):
            if predict[j] == 1 and temp_label[j] == 1:
                tp += 1
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
    print('the test score of epoch %d is %.4f' % (e, score / float(testNum)))
    y_axis.append(score / float(testNum))
    
    temp_Traindata = np.zeros(data.shape, dtype=data.dtype)
    temp_Trainlabel = np.zeros(label.shape, dtype=label.dtype)
    np.random.shuffle(order)
    for i in range(num):
        rd = np.random.randint(4)
        #data augmentation
        if rd == 0:
            temp_Traindata[i] = data[order[i]]
        if rd == 1:
            temp_Traindata[i] = data[order[i], :, ::-1]
        if rd == 2:
            temp_Traindata[i] = data[order[i], :, :, ::-1]
        if rd == 3:
            temp_Traindata[i] = data[order[i], :, ::-1, ::-1]
        temp_Traindata[i] = data[order[i]]
        temp_Trainlabel[i] = label[order[i]]
    #begin train  
    
    net.train()
    for i in range(batch):
        temp_data = temp_Traindata[i * miniBatch : i * miniBatch + miniBatch]
        temp_data = temp_data.astype(np.float32)
        temp_data = torch.from_numpy(temp_data)
        train_data = Variable(temp_data).cuda()
        temp_label = temp_Trainlabel[i * miniBatch : i * miniBatch + miniBatch]
        temp_label.resize(temp_label.shape[0], temp_label.shape[1])
        temp_label = temp_label.astype(np.float32)
        temp_label = torch. from_numpy(temp_label)
        train_label = Variable(temp_label).cuda()
        
        optimizer.zero_grad()
        outputs = net(train_data)
        #loss = Myloss(outputs, train_label)
        loss = F.binary_cross_entropy(outputs, train_label)
        loss.backward()
        optimizer.step()
        loss_accum += loss.data[0]
        
        if (i + 1) % show == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(e+1, epoch, i+1, batch, loss_accum / 20))
            loss_accum = 0.0
ax.plot(x_axis, y_axis)
plt.show()            
torch.save(net.state_dict(), './Kaggle_forest.pth')

#test
net.eval()
overall_array = np.zeros((len(category),), dtype=np.int32)
correct_array = np.zeros((len(category),), dtype=np.int32)
recall_array = np.zeros((len(category),), dtype=np.float32)
score = 0.0
for i in range(testNum):
    temp_data = val_data[i]
    temp_data = temp_data.reshape((1, channel, imgSize, imgSize))
    temp_data = temp_data.astype(np.float32)
    temp_data = torch.from_numpy(temp_data)
    test_data = Variable(temp_data).cuda()
    
    temp_label = val_label[i]
    output = net(test_data)
    
    output = output.cpu().data.numpy()[0]
    predict = np.zeros((len(category),), dtype=np.uint8)
    for j in range(len(category)):
        if output[j] >= 0.1:
            predict[j] = 1
    tp = 0
    fp = 0
    tf = 0
    ff = 0
    for j in range(len(category)):
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
print('the test score of epoch %d is %.4f' % (epoch + 1, score / float(testNum)))
for i in range(len(category)):
    recall_array[i] = correct_array[i] / float(overall_array[i])
print('finish')
