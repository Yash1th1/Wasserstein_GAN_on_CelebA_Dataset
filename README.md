# WGAN on CelebA Dataset

## Overview

This repository contains an implementation of a Wasserstein Generative Adversarial Network (WGAN) for generating images based on the CelebA dataset. The implementation leverages PyTorch and includes data preprocessing, model definition, training loop, and visualization of generated images.

## Requirements

- Python 3.7
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation

1. Install the required packages:

   ```bash
   pip install torch torchvision numpy matplotlib
   ```

2. Download the CelebA dataset:

   The dataset will be automatically downloaded when you run the training script. It will be stored in the `celebA/` directory.

## Model Architecture

### Generator

The generator network (`netg`) is a deep convolutional neural network that takes random noise as input and generates 64x64 RGB images. The architecture consists of several `ConvTranspose2d` layers with batch normalization and ReLU activations.

### Discriminator

The discriminator network (`netd`) is also a deep convolutional neural network that takes 64x64 RGB images as input and outputs a scalar value. The architecture consists of several `Conv2d` layers with batch normalization and LeakyReLU activations.

## Training

1. **Hyperparameters**: The hyperparameters are defined in the `Config` class. These include learning rate, noise dimension, image size, number of channels, batch size, and more.

2. **Data Preprocessing**: Images are resized to 64x64 and normalized.

3. **Training Loop**: The training loop consists of:
   - Clipping the discriminator's weights to avoid gradient explosion or vanishing.
   - Training the discriminator on both real and fake images.
   - Training the generator to produce more realistic images.
   - Visualizing the generated images after each epoch.

### Key Modifications for WGAN

- Removed the sigmoid activation in the last layer of the discriminator.
- Clipped the discriminator's parameters to a specified threshold.


### Results

Generated images are displayed after each epoch to monitor the progress of the generator. These images are produced using fixed noise vectors to ensure consistency.

### Acknowledgements

- [PyTorch](https://pytorch.org/)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

