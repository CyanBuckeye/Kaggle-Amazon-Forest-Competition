#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 20:38:05 2017

@author: andrew
input: single models' predictions
output: ensemble result
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

mydict = {'agriculture': 0,
 'artisinal_mine': 1,
 'bare_ground': 2,
 'blooming': 3,
 'blow_down': 4,
 'clear': 5,
 'cloudy': 6,
 'conventional_mine': 7,
 'cultivation': 8,
 'habitation': 9,
 'haze': 10,
 'partly_cloudy': 11,
 'primary': 12,
 'road': 13,
 'selective_logging': 14,
 'slash_burn': 15,
 'water': 16}

csvPath = [
'Res50_Final_TTA.csv',
'newRes50_TTA.csv',
'Res50_set1_TTA.csv',
'Res50_set2_TTA.csv',
'Res50_set2FC_TTA.csv',
'Res50_set4_TTA.csv',
'Res50_set3_TTA.csv'
]
#label predicition by single model
output_category = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 
            'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
csvfile_List = []
csvReader_List = []

for path in csvPath:
    f = open(path, 'r')
    csvfile_List.append(f)
    reader = csv.DictReader(f)
    csvReader_List.append(reader)

#in Kaggle competition, two test sets are given
testNumber=40669
fileNumber=20522
thresh = 4 #threshold for simple number voting.
voteCSVPath = 'vote.csv'#output csv   
f = open(voteCSVPath, 'w')
fieldnames = ['image_name', 'tags']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()

test_count = np.zeros((testNumber,17))
file_count = np.zeros((fileNumber,17))
for reader in csvReader_List:
    for row in reader:
        flag = False
        idx = row['image_name']
        if 'file' in idx:
            flag = True
        idx = int(idx.split('.')[0].split('_')[1])
        tags = row['tags']
        tags = tags.split(' ')
        for tag in tags:
            cat = mydict[tag]
            if flag == True:
                file_count[idx][cat] += 1
            else:
                test_count[idx][cat] += 1

for i in range(testNumber):
    print(i)
    total = 0
    predict = test_count[i]
    write_category = ""
    for j in range(len(predict)):
        if predict[j] >= thresh:
            total += 1
    for j in range(len(predict)):
        if predict[j] >= thresh:
            write_category += output_category[j]
            total -= 1
            if total > 0:
                write_category += ' '
    writer.writerow({'image_name': 'test_' + str(i), 'tags': write_category})                          

for i in range(fileNumber):
    print(i + testNumber)
    total = 0
    predict = file_count[i]
    write_category = ""
    for j in range(len(predict)):
        if predict[j] >= thresh:
            total += 1
    for j in range(len(predict)):
        if predict[j] >= thresh:
            write_category += output_category[j]
            total -= 1
            if total > 0:
                write_category += ' '
    writer.writerow({'image_name': 'file_' + str(i), 'tags': write_category})
f.close()

print('finish')