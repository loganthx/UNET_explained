# UNET_explained
We define our own UNET based on the original UNET article (source: https://arxiv.org/abs/1505.04597). It has less convolution layers in comparison to the reference (9 layers instead of 23), but the principle is the same.

# Dataset
MADS dataset is composed of 1192 images and their masks. We will use this dataset to train our UNET.
https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset


# Model Evaluation Methods
After training, we will evaluate the performance of the UNET on:
1. Inputs similar to MADS dataset
2. Inputs not similar to MADS dataset

# Training
We trained our model for 100 epochs, using an Adam optimizer with a learning rate of 0.0001. For loss calculations, we selected the Mean Squared Error metric.

![](https://i.ibb.co/sgbM9NY/Figure-4-log-100-epochs.png)

# Learning Geometrical Features
A UNET does this: 
1. Convolution until reaches a maximum of features (called bottleneck) while also max pooling after every convolution (reducing the size of the image)
2. Transpose2d and convolute without max pooling, decreasing the features and increasing the size until reaches the original image values. We have to apply connection between transpose2d layers and conv layers here, to make a so called decoder operation.

## UNET inputs, labels and outputs during training

![](https://i.ibb.co/qxHKcjv/TRAINING-GRAPH-EPOCHS30.png)

## Untrained Outputs

Considering the fact that a UNET is a good network for learning geometric characteristics of images, we should expect that even without training, the net should output images that, in a sense, maps geometric features of the input. 
The following images are outputs of the net without training at all.

![](https://i.ibb.co/TLFc4Gz/Untrained-Stimulus.png)

## Trained Outputs on Images Non Similar to Dataset

![](https://i.ibb.co/5LpVtrs/Trained-Stimulus.png)

## Trained Outputs on Images Similar to Dataset

![](https://i.ibb.co/1Ty4PWj/Unettesttwosamples.png)

# Future Works
Our main goal, since the beggining of this study, is understanding why denoising diffusion is so good at creating samples with high variety, according to ddpm article.
We seek to create a evaluation metric for this UNET in order to specify what is a reasonable minimum amount of layers in order to get a good result for well known image segmentation tasks. With that in mind, we will evaluate our UNET knowing for sure that it will be a good UNET for denoising diffusion applications.
