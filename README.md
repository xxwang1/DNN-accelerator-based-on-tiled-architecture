# Simulation framework of DNN accelerators based on tiled crossbar architecture
## Prerequisite packages
Numpy

Tensorflow
## Dataset
ImageNet

MNIST
## Functions
Layer_function: Fully Connected (FC) layer operation and Convolution (Conv) layer operation

Weight_mapping: Mapping FC weights and Conv weights onto crossbar arrays

Activation_mapping: Mapping FC activations and Conv activations into vectors

nonlinear: ReLu and ReLu6

img_TFRecords: Preprocessing for VGG

## Usage
Simulate VGG-16: run python Vgg.py

Simulate 4-array MNIST model: run python mnist_8bit.py

Simulate other models:
1. Define custom file, import numpy, weight_mapping, Layer_function and nonlinear;
2. In this file, load pretrained model;
3. Call weight_mapping functions to map weights into crossbar arrays;
4. Call Layer_function functions to build the network;
5. Call nonlinear functions to add activation functions between two layers.
