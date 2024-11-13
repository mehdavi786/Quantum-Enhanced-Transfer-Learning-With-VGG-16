# Quantum-Enhanced-Transfer-Learning-With-VGG-16

# VGG-16 Image Classification Model

VGG-16 is a convolutional neural network architecture developed by the Visual Geometry Group at the University of Oxford. It has been widely used in image recognition tasks due to its simple but highly effective structure. This README file provides an overview of the model, its architecture, and usage.

## Overview

The VGG-16 model is a 16-layer deep convolutional neural network primarily designed for image classification tasks. It has achieved state-of-the-art results on benchmarks like ImageNet, a large visual database designed for use in visual object recognition software research.

## Architecture

The VGG-16 model consists of 16 layers with learnable weights, specifically:
- **13 Convolutional Layers**: Using small 3x3 filters for capturing fine details.
- **3 Fully Connected Layers**: Used for dense feature extraction.
- **Max-Pooling Layers**: Each set of convolutional layers is followed by a max-pooling layer that reduces spatial dimensions.
- **ReLU Activation**: Each convolutional layer uses a ReLU activation function to introduce non-linearity.
- **Softmax Output Layer**: For classification tasks, the final layer is a softmax layer with 1000 outputs (for ImageNet).

The model is structured into blocks as follows:
1. **Block 1**: 2 convolutional layers followed by a max-pooling layer.
2. **Block 2**: 2 convolutional layers followed by a max-pooling layer.
3. **Block 3**: 3 convolutional layers followed by a max-pooling layer.
4. **Block 4**: 3 convolutional layers followed by a max-pooling layer.
5. **Block 5**: 3 convolutional layers followed by a max-pooling layer.
6. **Fully Connected Layers**: Three fully connected layers at the end with ReLU and softmax activations.

## Model Summary

```plaintext
Layer (type)              Output Shape              Param #
================================================================
Conv2D-1                   (None, 224, 224, 64)      1,792
Conv2D-2                   (None, 224, 224, 64)      36,928
MaxPooling2D-3             (None, 112, 112, 64)      0
Conv2D-4                   (None, 112, 112, 128)     73,856
Conv2D-5                   (None, 112, 112, 128)     147,584
MaxPooling2D-6             (None, 56, 56, 128)       0
...
Flatten-16                 (None, 25088)             0
Dense-17                   (None, 4096)              102,764,544
Dense-18                   (None, 4096)              16,781,312
Dense-19                   (None, 1000)              4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
