# Multiscale Patch-based Feature Graphs for Image Classification

Method based on Multiscale Patch-based Feature Graphs for Image Classification. Images present visual features in different scales in real-world scenarios, and most common approaches do not consider this fact. Our approach applies transfer learning for dealing with small datasets and leverages visual features extracted by pre-trained models from different scales. We based our approach on graph convolutional networks (GCN) that take graphs representing the images in different scales as input and whose nodes are characterized by features extracted by pre-trained models from regular image patches of different scales.

# Code files
-patchbased_graph.py
	
This is the code of our approach to generate graph of one single scale. We also apply the train and test of the classifier in this code. Use -h tag to see the help of the code.

-multiscale_graphs.py

This is the code of our approach to use 5 different scales (as in our best result). We also apply the train and test of the classifier in this code. Use -h tag to see the help of the code.

-simple_features.py
	
In this code, we use the approach based on simple features. We also apply the train and test of the classifier in this code.

-raw_image.py
	
This code uses the raw_image and train/test the model without transfer learning.

#Obs

All the codes have a dictionary of the models, enable to use the same that we apply in the paper experiments.

These codes specifically use the Stanford Dataset that can be downloaded in: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset or http://ai.stanford.edu/~jkrause/cars/car_dataset.html. We use three different .mat files to get the labels and the classes names: 'cars_test_annos_withlabels_eval.mat', 'cars_train_annos.mat' and 'cars_annos.mat'. Make sure you have the three files and the images in a 'stanford/' folder.

To use CLIP you need to download and install it from: https://github.com/openai/CLIP.

# Reference

For more information about the approach (pre-print): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352100
