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

