##Overview
This is our team's solution to Kaggle Competition of Understanding Amazon from Space (https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). My friend Mao helped me train and finetune the deep learning model. All of code was written by myself. Finally, we ranked 78 among 938 teams and won a bronze medal with final score of 0.92966 (F2-score). 

##About Contest
In competition, we were asked to label over 60000 images. Each image has one or more labels among the following seventeen ones: cloudy, partly_cloudy, haze, water, habitation, agriculture, road, cultivation, bare_ground, primary, slash_burn, selective_logging, blooming, conventional_mine, artisinal_mine, blow_down and clear.
![sample image](https://raw.githubusercontent.com/CyanBuckeye/Kaggle-Amazon-Forest-Competition/master/image/agriculture cultivation partly_cloudy primary road water.jpg)
Essentially it is a multi-label classification task. After exploring data, we found that there are three outstanding obstacles:
1. Huge Scale Variation. Take water as example: in some images, water is a brown trickle. But in some, water can take up the whole image.
2. Imbalance between classes. For example, there are over 35000 images labeled as "primary" among 40476 training samples while there are just less than 400 images with label of "selective_logging".
3. Limited training data. It is a challenge to get a well-trained deep neural network with around 40000 images. In contrast, shallow classifiers such as support vector machine, are less greedy on training data but suffer from inferior performance.

##Procedure
Our first idea was to design and implement deep neural network from scratch. To deal with problem 1, inspired by the deconvolution net for segmentation, we decided to use deconvolution network instead of classical convolution network; for problem 3, we utilized data augmentation to alleviate overfitting. And we tried to apply the same trick to solve the problem 2. However, network trained on balanced data performed worse. It can be explained that balanced data loses information of data distribution.

Although our first idea sounds reasonable, its performance is not good enough: F2-score on validation set is around 0.915. In addition, it took more than one day for training even we were equipped with GTX 1070 graphic card. Then we learnt from paper that finetuing may be a good solution. So we changed our policy: instead of training model from scratch, we imported pre-trained model, such as ResNet-50, and tuned its parameters on our dataset. I also managed to improve my code's efficiency. I noticed that the 32GB memory is able to accommodate the whole training set. So it is unnecessary to load each batch of data from disk then do data augmentation. I implemented data augmentation function and applied it on memory data. At last, training time was reduced to 5 hours from over 24 hours. What's more, we rented two P2 instances of AWS for further speedup.

"Where there is a will, there is a road". Through hard working and cooperation, my friend and I won a bronze medal. For me, there are much more valuable things I achieved from it: technique skills, programming experience, cooperation spirit and confidence with myself.         
