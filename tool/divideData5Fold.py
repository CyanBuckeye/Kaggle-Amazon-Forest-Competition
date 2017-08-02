#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:23:18 2017

@author: andrew
divide training samples into five parts evenly
the structure of data directory should be:
/data/class1/*.jpg
/data/class2/*.jpg
...

the structure of output directory will be:
/data/class1/0/*.jpg
/data/class1/1/*.jpg 
/data/class1/2/*.jpg
/data/class1/3/*.jpg
/data/class1/4/*.jpg
"""

import numpy as np
import os
from shutil import copy

category = ['cloudy', 'partly_cloudy', 'haze', 'clear']

inputPath = '' 
outputPath = ''           
for cat in category:
    inputPath = os.path.join(inputPath, cat)
    outputPath = os.path.join(outputPath, cat)
    if os.path.isdir(outputPath) == False:
        os.mkdir(outputPath)
    TotalNum = len(os.listdir(inputPath))
    valSize = int(0.2 * TotalNum)
    order = np.zeros((TotalNum,), dtype=np.int32)
    for idx, files in enumerate(os.listdir(inputPath)):
        file_idx = files.split('_')[-1]
        file_idx = file_idx.split('.')[0]
        order[idx] = file_idx
             
    np.random.shuffle(order)
    
    s = []
    for i in range(5):
        s.append([])
    s[0] = order[:valSize]
    s[1] = order[valSize:2*valSize]
    s[2] = order[2*valSize:3*valSize]
    s[3] = order[3*valSize:4*valSize]
    s[4] = order[4*valSize:]
    
    for idx, chooseList in enumerate(s):
        chooseList = list(chooseList)
        temp_outputPath = os.path.join(outputPath, str(idx))
        if os.path.isdir(temp_outputPath) == False:
            os.mkdir(temp_outputPath)
        for i in chooseList:
            fileName = 'train_' + str(i) + '.jpg'
            src = os.path.join(inputPath, fileName)
            dst = os.path.join(temp_outputPath, fileName)
            copy(src, dst)  
    