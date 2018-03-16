Kaggle Machine Learning Contest
===============================
Overview
-------------------------------
This is our team's solution to [Kaggle Competition of Understanding Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). My friend Mao helped me train and finetune the deep learning model. All of code was written by myself. Finally, we ranked 78 among 938 teams and won a bronze medal with final score of 0.92966 (F2-score). 

About Contest
-------------------------------
In competition, we were asked to label over 60000 images. Each image has one or more labels among the following seventeen ones: cloudy, partly_cloudy, haze, water, habitation, agriculture, road, cultivation, bare_ground, primary, slash_burn, selective_logging, blooming, conventional_mine, artisinal_mine, blow_down and clear.
Essentially it is a multi-label classification task.  
![sample image](https://raw.githubusercontent.com/CyanBuckeye/Kaggle-Amazon-Forest-Competition/master/image/sample.jpg)   
__An example image, whose corresponding label is agriculture cultivation partly_cloudy primary road water__

Difficulties
-------------------------------
1. Huge Scale Variation. Take water as example: in some images, water is a brown trickle. But in some, water can take up the whole image.
2. Imbalance between classes. For example, there are over 35000 images labeled as "primary" among 40476 training samples while there are just less than 400 images with label of "selective_logging".
3. Limited training data. It is a challenge to get a well-trained deep neural network with around 40000 images. In contrast, shallow classifiers such as support vector machine, are less greedy on training data but suffer from inferior performance.

How We Got Results
--------------------------------
### First method
design and implement deep neural network from scratch. To deal with problem 1, inspired by the deconvolution net for segmentation, we decided to use deconvolution network instead of classical convolution network; for problem 3, 
we utilized data augmentation to alleviate overfitting. And we tried to apply the same trick to solve the problem 2.  
![sample image](https://raw.githubusercontent.com/CyanBuckeye/Kaggle-Amazon-Forest-Competition/master/image/overfit_loss.jpg)
![sample image](https://raw.githubusercontent.com/CyanBuckeye/Kaggle-Amazon-Forest-Competition/master/image/overfit_score.jpg)


