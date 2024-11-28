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

```
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
```

# Applying Transfer Learning to VGG-16

Transfer learning is a technique that allows us to use the VGG-16 model, pre-trained on the ImageNet dataset, as the foundation for a new classification task. By reusing its learned features, we can adapt the model for our own dataset with significantly less training data and time.

## Steps in Transfer Learning with VGG-16

1. **Freeze Pre-Trained Layers:** The convolutional layers of VGG-16 capture valuable feature representations (such as edges, textures, and shapes). By freezing these layers, we preserve their weights, preventing them from updating during training on the new dataset. This keeps the core feature extraction ability of the model intact.
2. **Add Custom Classification Layers:** To adapt VGG-16 to a new classification task, we replace the original fully connected layers with new ones specific to our task:
- A flattening layer to transform convolutional output to a 1D vector.
- One or more dense layers, typically with ReLU activation, to learn task-specific patterns.
- A final output layer with a softmax or sigmoid activation function, depending on the task (multi-class or binary classification).
3. **Fine-Tuning the Model (Optional):** Additional performance improvements can be achieved by unfreezing a few deeper convolutional layers (near the output). Fine-tuning these layers allows them to adapt slightly to the new data.
4. **Compile and Train:** With the new layers added and optionally fine-tuned, we compile the model with an appropriate optimizer and loss function and then train it on the new dataset. Training is focused on the new layers (and optionally fine-tuned layers), which enables efficient learning without extensive data or training time.

# What are Quanvolutional Neural Networks (QNNs)?

Quanvolutional neural networks apply quantum circuits to classical input data, encoding features through quantum operations. Quantum circuits provide a unique transformation space that may capture intricate patterns, potentially leading to higher accuracy and robustness in certain classification tasks. In this project, QNNs are added to the last few layers of VGG-16, enabling a hybrid quantum-classical model for transfer learning.

## Quantum Circuit Layers
In this model:
- **Quantum Layers (Quanvolutional)**: Quantum circuits are applied to enhance features extracted from VGG-16’s convolutional layers.
- **Hybrid Quantum-Classical Layers**: After quantum processing, features are processed by a classical dense layer for final classification.

# Quantum-Inspired Hybrid Neural Network for Image Classification

This project implements a hybrid neural network combining **VGG16**, a pre-trained deep learning model, with a custom **Quantum-Inspired Feature Map** layer for advanced feature transformations. The code is structured to train and evaluate the model on the Intel Image Classification dataset.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Code Structure](#code-structure)
- [Model Architecture](#model-architecture)
- [Quantum-Inspired Feature Map](#quantum-inspired-feature-map)
- [Image Augmentation](#image-augmentation)
- [Learning Rate Scheduler](#learning-rate-scheduler)
- [Callbacks](#callbacks)

---

## Overview

This project showcases an innovative approach to image classification by integrating quantum-inspired techniques into a classical deep learning framework:
- The **VGG16 model** is used for feature extraction.
- A **custom feature transformation layer** inspired by quantum computing enhances the representational power of extracted features.
- Dense layers further refine the features, followed by a softmax output layer for classification.

The dataset used is the **Intel Image Classification Dataset**, containing six distinct classes.

---

## Features

### Key Features of the Code:
1. **Pre-trained Base Model**: The code uses VGG16 pre-trained on ImageNet, leveraging transfer learning to reduce training time and improve performance.
2. **Quantum-Inspired Feature Map Layer**: A custom Keras layer that applies trainable quantum-inspired transformations.
3. **Image Augmentation**: Extensive augmentation applied to training images to improve model robustness.
4. **Learning Rate Scheduler**: Dynamically adjusts learning rate based on training progress.
5. **Dropout Regularization**: Reduces overfitting by randomly deactivating neurons during training.

---

## Code Structure

### 1. **Imports and Libraries**:
The code imports necessary libraries like TensorFlow, Keras, NumPy, and image preprocessing tools. It also includes modules for creating and training the model.

### 2. **Input Parameters**:
The input size for images is `(224, 224, 3)`, matching VGG16's requirements. The number of classes is dynamically detected from the dataset.

### 3. **Custom Quantum-Inspired Layer**:
A trainable layer that transforms feature vectors with quantum-inspired non-linear operations.

### 4. **VGG16 Model Integration**:
The VGG16 model is loaded without the top dense layers, and its weights are frozen to retain pre-trained knowledge.

### 5. **Augmentation and Data Generators**:
The code applies transformations like rotation, zoom, shifts, and flips to enhance the diversity of the training data.

### 6. **Model Compilation**:
The model uses Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.

### 7. **Training**:
The training loop includes learning rate adjustment callbacks to ensure stable convergence.

---

## Model Architecture

The model architecture combines the strengths of a pre-trained model with a custom feature transformation layer. The architecture is as follows:

1. **Input Layer**: Accepts images of size `(224, 224, 3)`.
2. **VGG16 Backbone**: Extracts features from the input images.
3. **Flatten Layer**: Converts feature maps into a 1D vector.
4. **Dropout**: Reduces overfitting by randomly deactivating neurons.
5. **Quantum-Inspired Feature Map**:
   - Applies a trainable transformation inspired by quantum computing.
   - Uses sinusoidal activation to mimic quantum phase interactions.
6. **Dense Layers**: Further refines features with fully connected layers.
7. **Output Layer**: Produces class probabilities using softmax activation.

---

## Quantum-Inspired Feature Map

The **QuantumInspiredFeatureMap** layer applies a unique transformation:
- **Trainable Weights**: Each input feature is linearly transformed with weights.
- **Non-linear Activation**: A sine function is applied, mimicking quantum state evolution.
- **Normalization**: Ensures the output has normalized amplitudes.

This layer acts as a surrogate for quantum-inspired feature interactions.

---

## Image Augmentation

The code applies various augmentations to the input images to improve generalization:
- **Random Rotation**: Up to 40 degrees.
- **Zooming**: Random zoom in/out.
- **Shifting**: Horizontal and vertical shifts up to 20%.
- **Flipping**: Random horizontal flips.
- **Filling**: Uses nearest pixel values to fill gaps caused by transformations.

---

## Learning Rate Scheduler

A custom learning rate scheduler is implemented:
- **Decay Rate**: Multiplies the learning rate by 0.9 every 5 epochs.
- **Dynamic Adjustment**: Ensures smooth convergence and prevents oscillations.

This scheduler is passed as a callback to the training process.

---

## Callbacks

The model utilizes the following callback:
- **LearningRateScheduler**: Dynamically adjusts the learning rate based on the training epoch.

---

This file provides a detailed explanation of the functionality of the code and its modular structure.


## Advantages of Quantum Transfer Learning with VGG-16

- **Enhanced Feature Extraction**: Quantum circuits may capture unique features that improve classification accuracy.
- **Efficient Use of Data**: By leveraging quantum processing, the model can potentially generalize better on small datasets.
- **Reduced Training Time on Pre-Trained Layers**: Only the final quantum and classical layers require training, while pre-trained VGG-16 layers are kept frozen.
