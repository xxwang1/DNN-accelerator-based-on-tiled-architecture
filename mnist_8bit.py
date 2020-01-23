import numpy as np
import math
import Layer_function
from Layer_function import fc, fc_int, max_pool
from Layer_function import Conv2D, Conv2D_quant_uint, Conv2D_quant_int
import nonlinear
from nonlinear import batchnorm
from nonlinear import ReLu, Relu6_quant, Relu6

from weight_loading import load_from_tflite, load_from_pb
import Weight_mapping
from Weight_mapping import fc_mapping, fc_mapping_int
from Weight_mapping import conv_mapping, conv_mapping_int
import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set

import tensorflow as tf
import numpy as np
import os
import h5py
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from quant_train_layers import quant_train_Conv2D, quant_train_Conv, quant_Dense
import csv
#load model

sess = tf.Session()
with sess.as_default():
    model = tf.keras.models.load_model('4x64x64_chip_model/4x64x64_chip_model_8bit_act_weight.h5', custom_objects={'quant_train_Conv2D': quant_train_Conv2D, 'quant_Dense': quant_Dense})
    # model = tf.keras.models.load_model('4x64x64_chip_model/4x64x64_chip_model_8bit_act_weight.h5')

# model = torch.load('mnist_float_ckpt.t7')
# bias = {}
    weight = {}
    # bias['conv_0'] = model.weights[1]
    weight['conv_0'] = model.weights[0]
    # bias['conv_1'] = model.weights[3]
    weight['conv_1'] = model.weights[1]
    # bias['conv_2'] = model.weights[5]
    weight['conv_2'] = model.weights[2]
    # bias['fc'] = model.weights[7]
    weight['fc'] = model.weights[3]
    # bias['conv_0'] = bias['conv_0'].eval()
    weight['conv_0'] = weight['conv_0'].eval()
    # bias['conv_1'] = bias['conv_1'].eval()
    weight['conv_1'] = weight['conv_1'].eval()
    # bias['conv_2'] = bias['conv_2'].eval()
    weight['conv_2'] = weight['conv_2'].eval()
    # bias['fc'] = bias['fc'].eval()
    weight['fc'] = weight['fc'].eval()
# bias['conv_0'] = model['conv.0.bias'].numpy()
# weight['conv_0'] = model['conv.0.weight'].numpy()
# bias['conv_1'] = model['conv.3.bias'].numpy()
# weight['conv_1'] = model['conv.3.weight'].numpy()
# bias['conv_2'] = model['conv.6.bias'].numpy()
# weight['conv_2'] = model['conv.6.weight'].numpy()
# bias['fc'] = model['fc.0.bias'].numpy()
# weight['fc'] = model['fc.0.weight'].numpy()
# weight['conv_0'] = np.moveaxis(weight['conv_0'], -1, 0)
# weight['conv_1'] = np.moveaxis(weight['conv_2'], -1, 0)
# weight['conv_2'] = np.moveaxis(weight['conv_1'], -1, 0)
weight['fc'] = np.expand_dims(np.expand_dims(weight['fc'], axis=0), axis=0)

weight_scale = {}
weight_scale['conv_0'] = np.max((np.max(weight['conv_0']),  (-np.min(weight['conv_0'])))) / 127
weight_scale['conv_1'] = np.max((np.max(weight['conv_1']), (-np.min(weight['conv_1'])))) / 127
weight_scale['conv_2'] = np.max((np.max(weight['conv_2']),  (-np.min(weight['conv_2'])))) / 127
weight_scale['fc'] = np.max((np.max(weight['fc']), (-np.min(weight['fc'])))) / 127

weight['conv_0'] = np.round(weight['conv_0'] / weight_scale['conv_0'])
weight['conv_1'] = np.round(weight['conv_1'] / weight_scale['conv_1'])
weight['conv_2'] = np.round(weight['conv_2'] / weight_scale['conv_2'])
weight['fc'] = np.round(weight['fc'] / weight_scale['fc'])
# np.save('conv0', weight['conv_0'])
# np.save('conv1', weight['conv_1'])
# np.save('conv2', weight['conv_2'])
# np.save('fc', weight['fc'])

def MNIST_int(Pi, if_partial_ADC):
    #quantize weight
    # weight_scale = {}
    # weight_scale['conv_0'] = max(abs(np.max(weight['conv_0'])), abs(np.min(weight['conv_0']))) / 127
    # if weight_scale['conv_0'] == 0:
    #     weight_scale['conv_0'] = 0.001
    # weight_scale['conv_1'] = max(abs(np.max(weight['conv_1'])), abs(np.min(weight['conv_1'])) / 127)
    # if weight_scale['conv_1'] == 0:
    #     weight_scale['conv_1'] = 0.001
    # weight_scale['conv_2'] = max(abs(np.max(weight['conv_2'])), abs(np.min(weight['conv_2'])) / 127)
    # if weight_scale['conv_2'] == 0:
    #     weight_scale['conv_2'] = 0.001
    # weight_scale['fc'] = max(abs(np.max(weight['fc'])), abs(np.min(weight['fc'])) / 127)
    # if weight_scale['fc'] == 0:
    #     weight_scale['fc'] = 0.001
    # weight['conv_0'] = (weight['conv_0'] / weight_scale['conv_0'])
    # weight['conv_1'] = (weight['conv_1'] / weight_scale['conv_1'])
    # weight['conv_2'] = (weight['conv_2'] / weight_scale['conv_2'])
    # weight['fc'] = np.expand_dims(np.expand_dims((weight['fc'] / weight_scale['fc']), axis=0), axis=0)
    # weight['conv_0'] = np.moveaxis(np.moveaxis(weight['conv_0'], 1, -1), 0, -1)
    # weight['conv_1'] = np.moveaxis(np.moveaxis(weight['conv_1'], 1, -1), 0, -1)
    # weight['conv_2'] = np.moveaxis(np.moveaxis(weight['conv_2'], 1, -1), 0, -1)
    # weight['fc'] = np.expand_dims(np.expand_dims((weight['fc']), axis=0), axis=0)

    weight_mapped_0 = conv_mapping(weight['conv_0'])
    weight_mapped_1 = conv_mapping(weight['conv_1'])
    weight_mapped_2 = conv_mapping(weight['conv_2'])
    weight_mapped_fc = fc_mapping(weight['fc'])

    #read images
    ##############################################################################
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))])

    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                         shuffle=False)
    total = 0
    correct = 0
    for i in range(10000):
        images = np.moveaxis(test_images[i], -1, 0)
        # input_act = images.numpy()[0]
        # input_act = (input_act + 1)/2
        input_act = np.round(images * 255)
        
        #conv_0
        # bias_0 = bias['conv_0'] #/ weight_scale['conv_0'] * 127
        output_0 = Conv2D_quant_int(input_act, 1/255, weight['conv_0'], weight_mapped_0, weight_scale['conv_0'], np.zeros(22), 1, 1, 8, if_partial_ADC)
        # output_0 = Conv2D_quant_int(input_act, 1, weight['conv_0'], weight_mapped_0, 1, np.zeros(7), 1, 1, 8, False)
        input_1 = Relu6(output_0)
        input_1 = max_pool(input_1, 2)
        input_scale_1 = np.max(input_1) / 255
        input_1 = input_1 / input_scale_1
        # bias_1 = bias['conv_1'] #/ weight_scale['conv_1'] / np.max(input_1) * 255
        # input_1 = np.round(input_1 / np.max(input_1) * 255)
        #conv_1
        output_1 = Conv2D_quant_int(input_1, input_scale_1, weight['conv_1'], weight_mapped_1, weight_scale['conv_1'], np.zeros(27), 1, 1, 8, if_partial_ADC)
        # output_1 = Conv2D_quant_int(input_1, 1, weight['conv_1'], weight_mapped_1, 1, np.zeros(7), 1, 1, 8, False)
        input_2 = Relu6(output_1)
        input_2 = max_pool(input_2, 2)
        input_scale_2 = np.max(input_2) / 255
        input_2 = input_2 / input_scale_2
        # bias_2 = bias['conv_2'] #/ weight_scale['conv_2'] / np.max(input_2) * 255
        # input_2 = np.round(input_2 / np.max(input_2) * 255)
        #conv_1
        output_2 = Conv2D_quant_int(input_2, input_scale_2, weight['conv_2'], weight_mapped_2, weight_scale['conv_2'], np.zeros(64), 1, 2, 8, if_partial_ADC)
        # output_2 = Conv2D_quant_int(input_2, 1, weight['conv_2'], weight_mapped_2, 1, np.zeros(64), 1, 2, 8, False)
        input_3 = Relu6(output_2)
        input_3 = max_pool(input_3, 4)
        input_scale_3 = np.max(input_3) / 255
        input_3 = input_3 / input_scale_3
        # bias_3 = bias['fc'] #/ weight_scale['fc'] / np.max(input_3) * 255
        # input_3 = np.round(input_3 / np.max(input_3) * 255)
        #fc
        input_3 = np.expand_dims(np.squeeze(input_3), axis=0)
        output_fc = fc_int(input_3, input_scale_3, weight['fc'], weight_mapped_fc, weight_scale['fc'], np.zeros(10), 1, 8, if_partial_ADC)
        # output_fc = fc_int(input_3, 1, weight['fc'], weight_mapped_fc, 1, np.zeros(10), 1, 8, False)

        # print(output_fc.shape)
        f=open(Pi,'a+')
        print(np.max(output_fc), np.argmax(output_fc),file=f)
        f.close() #看一下列表里的内容

        del input_set[:]
        total += 1
        if np.argmax(output_fc) == test_labels[i]:
            correct += 1
        if total % 100  == 0:
            print("accuracy of " + str(total) + ' images is', correct / total)
        
        # del result
    # input_vector_set = np.array(input_vector_set)
    # np.save("input_vector_mnist_l3.npy", np.array(input_vector_set))
    # with open("input_vector_mnist_l3.csv", "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(input_vector_set)
    # pp_max = np.array(pp_max)
    # np.save("output_max_mnist_ls.npy", np.array(pp_max))
    # with open("output_max_mnist_ls.csv", "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(pp_max)

MNIST_int('8bit.txt', True)


