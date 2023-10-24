# GAN-boosted-Artificial-Neural-Networks
A vital component of this code is using a deep Artificial Neural Network (ANN) structure known as the Generative Adversarial Network (GAN). The GAN structure is combined with regressor (predictive) ANNs, to boost their performance when estimating the target parameter. 

# GAN-Boosted Artificial Neural Networks for ROP Estimation

This Jupyter Notebook contains code that implements an innovative approach for estimating the Rate of Drilling Bit Penetration (ROP) using deep learning techniques. The approach combines a deep Artificial Neural Network (ANN) structure known as the Generative Adversarial Network (GAN) with regressor (predictive) ANNs to enhance the accuracy of ROP estimation. 

## Overview

- The primary goal of this code is to develop a model for accurately estimating ROP, which is a critical parameter in drilling operations.
- The code leverages GANs, a deep learning architecture, to improve the performance of predictive ANNs.
- A novel feature of this code is the use of a residual structure during 1D-CNN (Convolutional Neural Network) training. This innovation enhances the 1D-CNN's performance by combining input data with features extracted from the inputs.

For a detailed explanation of the methodology and results, please refer to the associated research paper:

**Title:** Developing GAN-boosted Artificial Neural Networks to Model the Rate of Drilling Bit Penetration

**Published in:** Applied Soft Computing, 2023

**DOI:** [https://doi.org/10.1016/j.asoc.2023.110067](https://doi.org/10.1016/j.asoc.2023.110067)

## Requirements

Before running the notebook, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- NumPy
- TensorFlow
- Keras
- Matplotlib

You can install the required Python packages using pip with the following command:

```bash
pip install numpy tensorflow keras matplotlib


# Graphical Abstract

![2](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/d579165a-1c8f-4422-89a8-170f9fc8f08b)

# A general structure of 1D-CNN

![3](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/db6d3e2b-d2ea-4e49-a3c1-555a723aa748)

# Residual 1D-Convolutional Neural Network (Res 1D-CNN)

As part of this study, to improve the model’s performance, the general architecture of 1D-CNN is modified by adding a residual structure. However, the more complex and deep the ANN, the more likely it is to suffer from vanishing/exploding gradients, which results in performance degradation. He et al (http://dx.doi.org/10.1109/CVPR.2016.90) developed a deep residual architecture called ResNet to overcome this problem. It is a stack of residual blocks that makes up the ResNet architecture. A layer’s output is taken and added to the output of a deeper layer in the residual block. This crossing of layers in residual architecture is called skip connection or shortcutting. By introducing a residual form, the depth of the model can be increased without adding additional parameters to the training process or causing further computation complexity. The following figure illustrates the configuration of a single residual block and the skip connection. X, the output from the preceding layer, is used as the input for another layer (e.g., a convolutional block) where function F converts it to F(X). Following the transformation, the original data, X, is added to the transformed result, F(X)+X being the final result of the residual block.

![4](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/9d6f3e7e-9093-4d3b-99c0-9ee74bca1d07)

The following figure illustrates a schematic overview of the developed Res 1D-CNN architecture based on the discussed modifications. Throughout this study, the input layer is positioned next to the output of the 1D-CNN feature extraction section. As a result, the patterns in the input layer can remain intact alongside the extracted features. Thus, the learning section of 1D-CNN can use more information.

![5](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/b63b4187-2684-441f-abd8-6757a50156f4)

# Schematic representation of the GAN structure

consisting of a generator and a discriminator. The network receives a random matrix called the latent matrix
and creates a fake target sample similar to a true target sample.

![6](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/91ec7c1c-6e99-4dbc-827e-3bfb6ecbff4f)

# Schematic of an unboosted predictive ANN

This network can be MLP, 1D-CNN, and ResCNN, based on the hidden layers’ number, type, and structure.

![7](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/eccbbb27-93ec-451a-98c3-79fe984b60d0)

# Development of GAN-Boosted Neural Networks (GB-NNs)

The following figure shows schematic for a predictor ANN boosted by a pre-trained frozen GAN generator (GB-NN). The generator is frozen and placed into the predictive ANN in place of a single output neuron once the weights of the generator network are adjusted to produce fake ROP in the GAN training phase.

![8](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/4cea4c24-80f4-4bd2-8e44-b42a4e6b8f5f)

 # A flowchart of the training process of GB-NNs

![milad masrror article](https://github.com/miladmasroor/GAN-boosted-Artificial-Neural-Networks/assets/79324919/7d195f04-048b-45ef-95cd-13dad4604820)

