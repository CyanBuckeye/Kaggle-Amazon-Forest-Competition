This is our team's solution to Kaggle Amazon Forest Competition (https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).
We rank 78 among 938 teams and won a bronze model with final score of 0.92966 (F2-score). 

In competition, we are asked to label over 60000 images. Each image has one or more labels among the following seventeen ones: cloudy, partly_cloudy, haze, water, habitation, agriculture, road, cultivation, bare_ground, primary, slash_burn, selective_logging, blooming, conventional_mine, artisinal_mine, blow_down and clear. Essentially it is a multi-label classification task. After exploring data, we found that there are three outstanding obstacles:
1. Huge Scale Variation. Take water as example: in some images, water is a brown trickle. But in some, water can take the whole image.
2. Imbalance between classes. For example, there are over 35000 images with label of "primary" among 40476 training samples while there are just less than 400 images with label of "selective_logging".
3. Limited training data. It is a challenge to get a well-trained deep neural network with around 40000 images. In contrast, shallow classifiers such as support vector machine, are less greedy on training data but suffer from inferior performance.
