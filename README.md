# Quantum-Enhanced-Transfer-Learning-With-VGG-16

---

## Table of Contents
- [VGG-16 Image Classification Model](#VGG-16 Image Classification Model)
- [Overview](#overview)
- [Features](#features)
- [Code Structure](#code-structure)
- [Model Architecture](#model-architecture)
- [Quantum-Inspired Feature Map](#quantum-inspired-feature-map)
- [Image Augmentation](#image-augmentation)
- [Learning Rate Scheduler](#learning-rate-scheduler)
- [Callbacks](#callbacks)

---


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

```Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 224, 224, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ vgg16 (Functional)                   │ (None, 7, 7, 512)           │      14,714,688 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 25088)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 25088)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ quantum_inspired_feature_map         │ (None, 16)                  │         401,424 │
│ (QuantumInspiredFeatureMap)          │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │           2,176 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 6)                   │             774 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 15,119,062 (57.67 MB)
 Trainable params: 404,374 (1.54 MB)
 Non-trainable params: 14,714,688 (56.13 MB)
```

# Applying Transfer Learning to VGG-16

Transfer learning is a technique that allows us to use the VGG-16 model, pre-trained on the ImageNet dataset, as the foundation for a new classification task. By reusing its learned features, we can adapt the model for our own dataset with significantly less training data and time.

## Steps in Transfer Learning with VGG-16

1. **Freeze Pre-Trained Layers:** The convolutional layers of VGG-16 capture valuable feature representations (such as edges, textures, and shapes). By freezing these layers, we preserve their weights, preventing them from updating during training on the new dataset. This keeps the core feature extraction ability of the model intact.
2. **Add Custom Classification Layers:** To adapt VGG-16 to a new classification task, we replace the original fully connected layers with new ones specific to our task:
- A flattening layer to transform convolutional output to a 1D vector.
- One or more dense layers, typically with ReLU activation, to learn task-specific patterns.
- A final output layer with a softmax or sigmoid activation function, depending on the task (multi-class or binary classification).
3. **Quantum Inspired Feature Map:** A custom layer named QuantumInspiredFeatureMap is introduced to emulate quantum-inspired transformations:
- The layer includes trainable weights (kernel) and biases (bias) for learning complex feature interactions.
- A linear transformation is applied to the feature vector using the trainable kernel.
- A bias is added to the transformed features.
3. **Fine-Tuning the Model (Optional):** Additional performance improvements can be achieved by unfreezing a few deeper convolutional layers (near the output). Fine-tuning these layers allows them to adapt slightly to the new data.
4. **Compile and Train:** With the new layers added and optionally fine-tuned, we compile the model with an appropriate optimizer and loss function and then train it on the new dataset. Training is focused on the new layers (and optionally fine-tuned layers), which enables efficient learning without extensive data or training time.

# What are Quanvolutional Neural Networks (QNNs)?

Quanvolutional neural networks apply quantum circuits to classical input data, encoding features through quantum operations. Quantum circuits provide a unique transformation space that may capture intricate patterns, potentially leading to higher accuracy and robustness in certain classification tasks. In this project, QNNs are added to the last few layers of VGG-16, enabling a hybrid quantum-classical model for transfer learning.

## Quantum Circuit Layers
In this model:
- **Quantum Layers (Quanvolutional)**: Quantum circuits are applied to enhance features extracted from VGG-16’s convolutional layers.
- **Hybrid Quantum-Classical Layers**: After quantum processing, features are processed by a classical dense layer for final classification.

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

## Advantages of Quantum Transfer Learning with VGG-16

- **Enhanced Feature Extraction**: Quantum circuits may capture unique features that improve classification accuracy.
- **Efficient Use of Data**: By leveraging quantum processing, the model can potentially generalize better on small datasets.
- **Reduced Training Time on Pre-Trained Layers**: Only the final quantum and classical layers require training, while pre-trained VGG-16 layers are kept frozen.
