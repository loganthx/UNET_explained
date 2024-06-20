# UNET_explained
We define our own UNET based on the original UNET article. It has way less layers, but the principle is the same.


# Learning Geometrical Features
A UNET does this: 
1. Convolution until reaches a maximum of features (called bottleneck) while also max pooling after every convolution (reducing the size of the image)
2. Transpose2d and convolute without max pooling, decreasing the features and increasing the size until reaches the original image values. We have to apply connection between transpose2d layers and conv layers here, to make a so called decoder operation.

## What are the UNET inputs, labels and outputs, actually, in training:
![alt text](https://i.ibb.co/qxHKcjv/TRAINING-GRAPH-EPOCHS30.png)
