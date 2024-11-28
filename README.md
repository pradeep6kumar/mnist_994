# MNIST Low Parameter Model

This repository contains a PyTorch implementation of a low-parameter MNIST classifier that achieves >99% accuracy on the MNIST dataset.

## Model Overview

This model demonstrates exceptional performance with minimal parameters through:
- Efficient architecture design using only ~19.8K parameters
- Strategic use of batch normalization and dropout for regularization
- Achieving 99.54% accuracy without complex architectures
- Fast convergence, reaching >99% accuracy within 15 epochs
- Stable training with consistent performance

## Model Architecture (Summary)

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             416
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 16, 24, 24]           6,416
         MaxPool2d-4           [-1, 16, 12, 12]               0
           Dropout-5           [-1, 16, 12, 12]               0
            Conv2d-6           [-1, 16, 12, 12]           6,416
       BatchNorm2d-7           [-1, 16, 12, 12]              32
            Conv2d-8             [-1, 16, 8, 8]           6,416
         MaxPool2d-9             [-1, 16, 4, 4]               0
          Dropout-10             [-1, 16, 4, 4]               0
AdaptiveAvgPool2d-11             [-1, 16, 1, 1]               0
           Linear-12                   [-1, 10]             170
================================================================
Total params: 19,898
Trainable params: 19,898
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.34
Params size (MB): 0.08
Estimated Total Size (MB): 0.42
----------------------------------------------------------------
```

## Dataset Description

The MNIST dataset is a widely used benchmark in machine learning consisting of handwritten digits:
- **Training Set**: 60,000 handwritten digit images
- **Test Set**: 10,000 handwritten digit images
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)
- **Image Type**: Grayscale, normalized (0-1)

### Dataset Features
- Clean, preprocessed images
- Balanced class distribution
- Standard benchmark for deep learning models
- Suitable for testing model architectures and optimization techniques

## Model Performance

The model achieved a final best test accuracy of **99.54%** during training, with multiple accuracy improvements throughout the training process.

### Best Model Checkpoints

The model showed consistent improvement, reaching new best accuracies at the following epochs:
1. Epoch 1: 97.84% - Initial checkpoint
2. Epoch 2: 98.54% - +0.70% improvement
3. Epoch 4: 98.99% - +0.45% improvement
4. Epoch 5: 99.20% - +0.21% improvement
5. Epoch 10: 99.39% - +0.19% improvement
6. Epoch 15: 99.43% - +0.04% improvement
7. Epoch 16: **99.54%** - Final best (+0.11% improvement)

### Training Analysis

- **Initial Performance**: Started with a high loss of 2.280499, quickly improved to sub-0.1 loss within the first epoch
- **Convergence**: Achieved >99% accuracy by epoch 5
- **Best Performance**: Reached peak accuracy of 99.54% at epoch 16
- **Final Loss**: Maintained stable low loss values (<0.02) in later epochs
- **Consistency**: Maintained >99% accuracy from epoch 5 onwards

## For a full log you can click below
![log](https://github.com/pradeep6kumar/mnist_994/blob/aoc/model_test_results_20241128_192148.txt)

## Epoch wise loss and accuracy

![plot](https://github.com/pradeep6kumar/mnist_994/blob/aoc/training_metrics.png)
