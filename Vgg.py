import numpy as np
import math
import Layer_function
from Layer_function import fc, fc_int, max_pool
from Layer_function import Conv2D, Conv2D_quant_uint, Conv2D_quant_int
import nonlinear
from nonlinear import batchnorm
from nonlinear import ReLu, Relu6_quant
import img_TFRecords
from img_TFRecords import image_to_act_vgg

from weight_loading import load_from_tflite, load_from_pb
import Weight_mapping
from Weight_mapping import fc_mapping, fc_mapping_int
from Weight_mapping import conv_mapping, conv_mapping_int
import gl
from gl import cb_size, scale_mode, latency, energy
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set, scale2bit, array2bit

import tensorflow as tf
import os
import h5py
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from ckpt import weight, bias
import multiprocessing as mp 

#os.system('taskset -p 0xffffffff %d' % os.getpid())

def VGG_16(start_img, end_img, act_bit, ADC_bit, Pi, ADC):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    # global weight
    weight_scale = {}
    weight_sum = []
    cb_count = []
    crossbar_sum = 0
    weight_layer = []
    for i in range(0,13):
        weight_sum.append(0)
        weight_scale['conv_' + str(i)] = (max(abs(np.max(weight['conv_' + str(i)])), abs(np.min(weight['conv_' + str(i)]))) / 127)
        if weight_scale['conv_' + str(i)] == 0:
             weight_scale['conv_' + str(i)] = (0.001)
        weight['conv_' + str(i)] = np.round(weight['conv_' + str(i)] / weight_scale['conv_' + str(i)])
        
        # weight_sum += np.sum(np.absolute(weight['conv_' + str(i)]))
        locals()['weight_conv_'+str(i)+'_mapped'] = conv_mapping(weight['conv_' + str(i)])
        cb_count.append(locals()['weight_conv_'+str(i)+'_mapped'].shape[0] * locals()['weight_conv_'+str(i)+'_mapped'].shape[1])
        weight_sum_layer = 0
        for j in range(len(locals()['weight_conv_'+str(i)+'_mapped'][0])):#number of arrays per row
            for k in range(len(locals()['weight_conv_'+str(i)+'_mapped'])):#number of arrays per column
                weight1 = crossbar_set[int(locals()['weight_conv_'+str(i)+'_mapped'][k][j])]#.get_array() 
                weight_sum_layer += np.sum(weight1.get_array())
        weight_layer.append(weight_sum_layer)
        #         for m in range((weight1.weight_scale[0]).shape[0]):
        # # #             if (weight1.weight_scale)[m] > weight_sum[-1]:
        # # #                 weight_sum[-1] = (weight1.weight_scale)[m]
        #             if (weight1.weight_scale[0])[m] !=0: #>= 655.35:
        #                 weight_sum.append(array2bit(weight1.weight_scale[0])[m])
        #             if (weight1.weight_scale[1])[m] !=0: #>= 655.35:
        #                 weight_sum.append(array2bit(weight1.weight_scale[1])[m])
        # if scale_mode == 'per_layer':
        #     for j in range(len(locals()['weight_conv_'+str(i)+'_mapped'][0])):#number of arrays per row
        #         for k in range(len(locals()['weight_conv_'+str(i)+'_mapped'])):#number of arrays per column
        #             crossbar_set[int(locals()['weight_conv_'+str(i)+'_mapped'][k][j])].weight_scale = np.ones(cb_size[1]) * weight_sum[-1] 
        # w_shape = locals()['weight_conv_'+str(i)+'_mapped'].shape
        # crossbar_sum += w_shape[0] * w_shape[1]
        # print('conv_' + str(i), w_shape[0] * w_shape[1])
    weight_sum.append(0)
    weight_scale['fc_13'] = (max(abs(np.max(weight['fc_13'])), abs(np.min(weight['fc_13']))) / 127)
    if weight_scale['fc_13'] == 0:
        weight_scale['fc_13'] = (0.001)
    weight['fc_13'] = np.round(weight['fc_13'] / weight_scale['fc_13'])
    # weight_sum += np.sum(np.absolute(weight['fc_13']))
    weight_sum_layer = 0
    locals()['weight_fc_13_mapped'] = conv_mapping(weight['fc_13'])
    cb_count.append(locals()['weight_fc_13_mapped'].shape[0] * locals()['weight_fc_13_mapped'].shape[1])
    for j in range(len(locals()['weight_fc_13_mapped'][0])):#number of arrays per row
        for k in range(len(locals()['weight_fc_13_mapped'])):#number of arrays per column
            weight1 = crossbar_set[int(locals()['weight_fc_13_mapped'][k][j])]#.get_array() 
            weight_sum_layer += np.sum(weight1.get_array())
    weight_layer.append(weight_sum_layer)
#             for m in range((weight1.weight_scale[0]).shape[0]):
# # #                 if (weight1.weight_scale)[m] > weight_sum[-1]:
# # #                     weight_sum[-1] = (weight1.weight_scale)[m]
#                 if (weight1.weight_scale[0])[m] !=0: #>= 655.35:
#                     weight_sum.append(array2bit(weight1.weight_scale[0])[m])
#                 if (weight1.weight_scale[1])[m] !=0: #>= 655.35:
#                     weight_sum.append(array2bit(weight1.weight_scale[1])[m])
#     if scale_mode == 'per_layer':
#         for j in range(len(locals()['weight_fc_13_mapped'][0])):#number of arrays per row
#             for k in range(len(locals()['weight_fc_13_mapped'])):#number of arrays per column
#                 crossbar_set[int(locals()['weight_fc_13_mapped'][k][j])].weight_scale = np.ones(cb_size[1]) * weight_sum[-1] 
    # w_shape = locals()['weight_fc_13_mapped'].shape
    # crossbar_sum += w_shape[0] * w_shape[1]
    # print('fc_13', w_shape[0] * w_shape[1])
    for i in range(14, 16):
        weight_sum.append(0)
        weight_scale['fc_' + str(i)] = (max(abs(np.max(weight['fc_' + str(i)])), abs(np.min(weight['fc_' + str(i)]))) / 127)
        if weight_scale['fc_' + str(i)] == 0:
            weight_scale['fc_' + str(i)] = (0.001)
        weight['fc_' + str(i)] = np.round(weight['fc_' + str(i)] / weight_scale['fc_' + str(i)])
        # weight_sum += np.sum(np.absolute(weight['fc_' + str(i)]))
        locals()['weight_fc_'+str(i)+'_mapped'] = fc_mapping(weight['fc_' + str(i)])
        cb_count.append(locals()['weight_fc_'+str(i)+'_mapped'].shape[0] * locals()['weight_fc_'+str(i)+'_mapped'].shape[1])
        weight_sum_layer = 0
        for j in range(len(locals()['weight_fc_'+str(i)+'_mapped'][0])):#number of arrays per row
            for k in range(len(locals()['weight_fc_'+str(i)+'_mapped'])):#number of arrays per column
                weight1 = crossbar_set[int(locals()['weight_fc_'+str(i)+'_mapped'][k][j])]#.get_array() 
                weight_sum_layer += np.sum(weight1.get_array())
        weight_layer.append(weight_sum_layer)
        #         for m in range((weight1.weight_scale[0]).shape[0]):
        # # #             if (weight1.weight_scale)[m] > weight_sum[-1]:
        # # #                 weight_sum[-1] = (weight1.weight_scale)[m]
        #             if (weight1.weight_scale[0])[m] !=0: #>= 655.35:
        #                 weight_sum.append(array2bit(weight1.weight_scale[0])[m])
        #             if (weight1.weight_scale[1])[m] !=0: #>= 655.35:
        #                 weight_sum.append(array2bit(weight1.weight_scale[1])[m])
        # if scale_mode == 'per_layer':
        #     for j in range(len(locals()['weight_fc_'+str(i)+'_mapped'][0])):#number of arrays per row
        #         for k in range(len(locals()['weight_fc_'+str(i)+'_mapped'])):#number of arrays per column
        #             crossbar_set[int(locals()['weight_fc_'+str(i)+'_mapped'][k][j])].weight_scale = np.ones(cb_size[1]) * weight_sum[-1] 
    #     w_shape = locals()['weight_fc_'+str(i)+'_mapped'].shape
    #     print('fc_' + str(i), w_shape[0] * w_shape[1])
    #     crossbar_sum += w_shape[0] * w_shape[1]
    # print('weight_sum=', weight_sum)
    # print('crossbar_sum=', crossbar_sum)
    #read images
    # weight_scale_array = np.histogram(array2bit(np.array(weight_sum)), bins='auto')
    # np.savetxt('vgg_4scale.txt', weight_scale_array, fmt='%s')
    # matplotlib.rcParams.update({'font.size': 20})
    # plt.hist(np.array(weight_sum), bins=100)
    # plt.xlabel('Partial sum scale', {'size':20})
    # plt.ylabel('Count', {'size':20})
    # ax = plt.gca()
    # ax.spines['right'].set_linewidth(3)
    # ax.spines['left'].set_linewidth(3)
    # ax.spines['top'].set_linewidth(3)
    # ax.spines['bottom'].set_linewidth(3)
    # xfmt = ScalarFormatter(useMathText=True)
    # xfmt.set_powerlimits((0, 0))
    # ax.yaxis.set_major_formatter(xfmt)
    # plt.tick_params(labelsize=16)

    # plt.show()
    # with open('vgg_weight_sum.txt', 'w') as f:
    #     for item in weight_layer:
    #         f.write("%s\n" % item)
    ##############################################################################
    path ="../ILSVRC2012_img_val/"    #指定需要读取文件的目录
    files =os.listdir(path) #采用listdir来读取所有文件
    files.sort() #排序
    #def MobileNet(start_img, end_img):
    # energy = 0
    for i in range(start_img,end_img):
        #os.system('taskset -p 0xffffffff %d' % os.getpid())
        # energy.append(0)
        file_=files[i]
        if not os.path.isdir(path +file_):#判断该文件是否是一个文件夹
            f_name = str(file_)
            print(f_name)
            input_act=image_to_act_vgg(f_name)
            input_scale = max(abs(np.max(input_act)), abs(np.min(input_act)))
            input_act = np.round(input_act/ scale2bit(input_scale / (2**(act_bit-1)-1)))
            
            #conv1_1
            input_pos = ReLu(input_act)
            input_act=image_to_act_vgg(f_name)
            input_act = np.round(input_act)
            input_neg = ReLu(input_act * (-1))
            bias_conv_0 = (bias['conv_0'] / weight_scale['conv_0'] / scale2bit(input_scale / (2**(act_bit-1)-1)))
            # print('bias_conv_0', bias_conv_0)
            output_conv0 = Conv2D_quant_int(input_pos, 1, weight['conv_0'], locals()['weight_conv_0_mapped'], 1, bias_conv_0, 1, 1, ADC_bit, ADC) - Conv2D_quant_int(input_neg, 1, weight['conv_0'], locals()['weight_conv_0_mapped'], 1, np.zeros_like(bias['conv_0']), 1, 1, ADC_bit, ADC)
            input_conv1 = ReLu(output_conv0)
            
            #conv1_2
            # print('input_conv1_max=', np.max(input_conv1))
            bias_conv_1 = (bias['conv_1'] / weight_scale['conv_1'] / scale2bit(np.max(input_conv1) / (2**act_bit-1)))
            # print('bias_conv_1', bias_conv_1)
            # print('input_conv1_max=', np.max(input_conv1))
            input_conv1 = np.round(input_conv1 / scale2bit(np.max(input_conv1) / (2**act_bit-1)))
            output_conv1 = Conv2D_quant_int(input_conv1, 1, weight['conv_1'], locals()['weight_conv_1_mapped'], 1, bias_conv_1, 1, 1, ADC_bit, ADC)
            input_conv2 = ReLu(output_conv1)
            
            #max pool
            bias_conv_2 = (bias['conv_2'] / weight_scale['conv_2'] / scale2bit(np.max(input_conv2) / (2**act_bit-1)))
            input_conv2 = np.round(input_conv2 / scale2bit(np.max(input_conv2) / (2**act_bit-1)))
            input_conv2 = max_pool(input_conv2, 2)
            # input_conv2 = np.expand_dims(np.moveaxis(input_conv2, 0, -1), axis=0)
            # input_conv2 = tf.convert_to_tensor(input_conv2)
            # input_conv2 = tf.layers.max_pooling2d(input_conv2, (2,2), (2,2), 'valid', data_format='channels_last')
            # with tf.Session() as Sess:
            #     input_conv2=Sess.run(input_conv2)
            # input_conv2=np.squeeze(np.moveaxis(input_conv2, -1, 0))

            #conv2_1
            
            output_conv2 = Conv2D_quant_int(input_conv2, 1, weight['conv_2'], locals()['weight_conv_2_mapped'], 1, bias_conv_2, 1, 1, ADC_bit, ADC)
            input_conv3 = ReLu(output_conv2)
            
            #conv2_2
            bias_conv_3 = (bias['conv_3'] / weight_scale['conv_3'] / scale2bit(np.max(input_conv3) / (2**act_bit-1)))
            input_conv3 = np.round(input_conv3 / scale2bit(np.max(input_conv3) / (2**act_bit-1)))
            output_conv3 = Conv2D_quant_int(input_conv3, 1, weight['conv_3'], locals()['weight_conv_3_mapped'], 1, bias_conv_3, 1, 1, ADC_bit, ADC)
            input_conv4 = ReLu(output_conv3)
            
            #max pool
            bias_conv_4 = (bias['conv_4'] / weight_scale['conv_4'] / scale2bit(np.max(input_conv4) / (2**act_bit-1)))
            input_conv4 = np.round(input_conv4 / scale2bit(np.max(input_conv4) / (2**act_bit-1)))
            input_conv4 = max_pool(input_conv4, 2)
            # input_conv4 = np.expand_dims(np.moveaxis(input_conv4, 0, -1), axis=0)
            # input_conv4 = tf.convert_to_tensor(input_conv4)
            # input_conv4 = tf.layers.max_pooling2d(input_conv4, (2,2), (2,2), 'valid', data_format='channels_last')
            # with tf.Session() as Sess:
            #     input_conv4=Sess.run(input_conv4)
            # input_conv4=np.squeeze(np.moveaxis(input_conv4, 0, -1))

            #conv3_1
            
            output_conv4 = Conv2D_quant_int(input_conv4, 1, weight['conv_4'], locals()['weight_conv_4_mapped'], 1, bias_conv_4, 1, 1, ADC_bit, ADC)
            input_conv5 = ReLu(output_conv4)
            
            
            #conv3_2
            bias_conv_5 = (bias['conv_5'] / weight_scale['conv_5'] / scale2bit(np.max(input_conv5) / (2**act_bit-1)))
            input_conv5 = np.round(input_conv5 / scale2bit(np.max(input_conv5) / (2**act_bit-1)))
            output_conv5 = Conv2D_quant_int(input_conv5, 1, weight['conv_5'], locals()['weight_conv_5_mapped'], 1, bias_conv_5, 1, 1, ADC_bit, ADC)
            input_conv6 = ReLu(output_conv5)
            
            #conv3_3
            bias_conv_6 = (bias['conv_6'] / weight_scale['conv_6'] / scale2bit(np.max(input_conv6) / (2**act_bit-1)))
            input_conv6 = np.round(input_conv6 / scale2bit(np.max(input_conv6) / (2**act_bit-1)))
            output_conv6 = Conv2D_quant_int(input_conv6, 1, weight['conv_6'], locals()['weight_conv_6_mapped'], 1, bias_conv_6, 1, 1, ADC_bit, ADC)
            input_conv7 = ReLu(output_conv6)
            
            #max pool
            bias_conv_7 = (bias['conv_7'] / weight_scale['conv_7'] / scale2bit(np.max(input_conv7) / (2**act_bit-1)))
            input_conv7 = np.round(input_conv7 / scale2bit(np.max(input_conv7) / (2**act_bit-1)))
            input_conv7 = max_pool(input_conv7, 2)
            # input_conv127 = np.expand_dims(np.moveaxis(input_conv127, 0, -1), axis=0)
            # input_conv127 = tf.convert_to_tensor(input_conv127)
            # input_conv127 = tf.nn.max_pool(input_conv127, (1,2,2,1), (1,2,2,1), 'VALID', "NHWC")
            # with tf.Session() as Sess:
            #     input_conv127=Sess.run(input_conv127)
            # input_conv127=np.squeeze(np.moveaxis(input_conv127, 0, -1))

            #conv4_1
            
            output_conv7 = Conv2D_quant_int(input_conv7, 1, weight['conv_7'], locals()['weight_conv_7_mapped'], 1, bias_conv_7, 1, 1, ADC_bit, ADC)
            input_conv8 = ReLu(output_conv7)
            

            #conv4_2
            bias_conv_8 = (bias['conv_8'] / weight_scale['conv_8'] / scale2bit(np.max(input_conv8) / (2**act_bit-1)))
            input_conv8 = np.round(input_conv8 / scale2bit(np.max(input_conv8) / (2**act_bit-1)))
            output_conv8 = Conv2D_quant_int(input_conv8, 1, weight['conv_8'], locals()['weight_conv_8_mapped'], 1, bias_conv_8, 1, 1, ADC_bit, ADC)
            input_conv9 = ReLu(output_conv8)
            
            #conv4_3
            bias_conv_9 = (bias['conv_9'] / weight_scale['conv_9'] / scale2bit(np.max(input_conv9) / (2**act_bit-1)))
            input_conv9 = np.round(input_conv9 / scale2bit(np.max(input_conv9) / (2**act_bit-1)))
            output_conv9 = Conv2D_quant_int(input_conv9, 1, weight['conv_9'], locals()['weight_conv_9_mapped'], 1, bias_conv_9, 1, 1, ADC_bit, ADC)
            input_conv10 = ReLu(output_conv9)
            
            #max pool
            bias_conv_10 = (bias['conv_10'] / weight_scale['conv_10'] / scale2bit(np.max(input_conv10) / (2**act_bit-1)))
            input_conv10 = np.round(input_conv10 / scale2bit(np.max(input_conv10) / (2**act_bit-1)))
            input_conv10 = max_pool(input_conv10, 2)
            # input_conv10 = np.expand_dims(np.moveaxis(input_conv10, 0, -1), axis=0)
            # input_conv10 = tf.convert_to_tensor(input_conv10)
            # input_conv10 = tf.nn.max_pool(input_conv10, (1,2,2,1), (1,2,2,1), 'VALID', "NHWC")
            # with tf.Session() as Sess:
            #     input_conv10=Sess.run(input_conv10)
            # input_conv10=np.squeeze(np.moveaxis(input_conv10, 0, -1))

            #conv5_1
            
            output_conv10 = Conv2D_quant_int(input_conv10, 1, weight['conv_10'], locals()['weight_conv_10_mapped'], 1, bias_conv_10, 1, 1, ADC_bit, ADC)
            input_conv11 = ReLu(output_conv10)
            
            #conv5_2
            bias_conv_11 = (bias['conv_11'] / weight_scale['conv_11'] / scale2bit(np.max(input_conv11) / (2**act_bit-1)))
            input_conv11 = np.round(input_conv11 / scale2bit(np.max(input_conv11) / (2**act_bit-1)))
            output_conv11 = Conv2D_quant_int(input_conv11, 1, weight['conv_11'], locals()['weight_conv_11_mapped'], 1, bias_conv_11, 1, 1, ADC_bit, ADC)
            input_conv12 = ReLu(output_conv11)
            
            #conv5_3
            bias_conv_12 = (bias['conv_12'] / weight_scale['conv_12'] / scale2bit(np.max(input_conv12) / (2**act_bit-1)))
            input_conv12 = np.round(input_conv12 / scale2bit(np.max(input_conv12) / (2**act_bit-1)))
            output_conv12 = Conv2D_quant_int(input_conv12, 1, weight['conv_12'], locals()['weight_conv_12_mapped'], 1, bias_conv_12, 1, 1, ADC_bit, ADC)
            input_conv13 = ReLu(output_conv12)
            
            #max pool
            bias_fc_13 = (bias['fc_13'] / weight_scale['fc_13'] / scale2bit(np.max(input_conv13) / (2**act_bit-1)))
            input_conv13 = np.round(input_conv13 / scale2bit(np.max(input_conv13) / (2**act_bit-1)))
            input_conv13 = max_pool(input_conv13, 2)
            # input_conv13 = np.expand_dims(np.moveaxis(input_conv13, 0, -1), axis=0)
            # input_conv13 = tf.convert_to_tensor(input_conv13)
            # input_conv13 = tf.nn.max_pool(input_conv13, (1,2,2,1), (1,2,2,1), 'VALID', "NHWC")
            # with tf.Session() as Sess:
            #     input_conv13=Sess.run(input_conv13)
            # input_conv13=np.squeeze(np.moveaxis(input_conv13, 0, -1))
            
            #fc1
            
            output_fc14 = Conv2D_quant_int(input_conv13, 1, weight['fc_13'], locals()['weight_fc_13_mapped'], 1, bias_fc_13, 1, 7, ADC_bit, ADC)
            output_fc14 = ReLu(output_fc14)
            
            #fc2
            output_fc14 = output_fc14.reshape(4096)
            output_fc14 = np.expand_dims(output_fc14, axis=0)
            bias_fc_14 = (bias['fc_14'] / weight_scale['fc_14'] / scale2bit(np.max(output_fc14) / (2**act_bit-1)))
            output_fc14 = np.round(output_fc14 / scale2bit(np.max(output_fc14) / (2**act_bit-1)))
            output_fc15  = fc_int(output_fc14, 1, weight['fc_14'], locals()['weight_fc_14_mapped'], 1, bias_fc_14, 1, ADC_bit, ADC)
            output_fc15 = ReLu(output_fc15)
            
            #fc3
            output_fc15 = output_fc15.reshape(4096)
            output_fc15 = np.expand_dims(output_fc15, axis=0)
            bias_fc_15 = (bias['fc_15'] / weight_scale['fc_15'] / scale2bit(np.max(output_fc15) / (2**act_bit-1)))
            output_fc15 = np.round(output_fc15 / scale2bit(np.max(output_fc15) / (2**act_bit-1)))
            output_fc16 = fc_int(output_fc15, 1, weight['fc_15'], locals()['weight_fc_15_mapped'], 1, bias_fc_15, 1, ADC_bit, ADC)

            output_fc = output_fc16

            print(output_fc.shape)
            f=open(Pi,'a+')
            f.write(f_name + ' ') # 写入之前的文本中
            print(np.max(output_fc), np.argmax(output_fc),file=f)
            f.close() #看一下列表里的内容

            del input_set[:]
            with open('vgg_latency.txt', 'w') as f:
                for item in latency:
                    f.write("%s\n" % item)
            # del result

# #os.system('taskset -p 0xffffffff %d' % os.getpid())
VGG_16(0, 100, 8, 6, 'vgg_test.txt', True)
# if __name__ == '__main__':
# #     print('number of cpu=', os.cpu_count())
# #     print('\n\n\n\n\n')
#     for k in range(os.cpu_count()):
#         i = k + 48
#         # if i<2:
#         #     P = mp.Process(target=VGG_16, args=(i*521+312, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         # if i == 3:
#         #     P = mp.Process(target=VGG_16, args=(i*521+92, (i)*521+202, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         # if i == 4:
#         #     P = mp.Process(target=VGG_16, args=(i*521+58, (i)*521+202, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         # if i==5:
#         #     P = mp.Process(target=VGG_16, args=(i*521+66, (i)*521+202, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         # if i == 7:
#         #     P = mp.Process(target=VGG_16, args=(i*521+67, (i)*521+202, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         # if i == 9:
#         #     P = mp.Process(target=VGG_16, args=(i*521+111, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         if i == 12 or i == 15 or i == 20 or i == 70:
#             P = mp.Process(target=VGG_16, args=(i*521+113, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         # if i == 21:
#         #     P = mp.Process(target=VGG_16, args=(i*521+183, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         if i == 22 or i == 26 or i == 30 or i == 33 or i == 68:
#             P = mp.Process(target=VGG_16, args=(i*521+203, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         # # if i == 27 or i == 28 or i == 31 or i == 32 or i == 34 or i == 36 or i == 37:
#         # #     P = mp.Process(target=VGG_16, args=(i*521+261, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         # #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         # #     P.start()
#         # # if i>=38 and i<=45:
#         # #     P = mp.Process(target=VGG_16, args=(i*521+261, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#         # #     os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#         #     P.start()
#         if i == 47 or i == 53 or i == 52 or i == 55:
#             P = mp.Process(target=VGG_16, args=(i*521+261, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         if i == 64:
#             P = mp.Process(target=VGG_16, args=(i*521+314, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         if i == 71:
#             P = mp.Process(target=VGG_16, args=(i*521+289, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         if i == 73:
#             P = mp.Process(target=VGG_16, args=(i*521+239, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         if i == 75 or i>=79:
#             P = mp.Process(target=VGG_16, args=(i*521+58, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
#         if i == 76:
#             P = mp.Process(target=VGG_16, args=(i*521+368, (i+1)*521, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_dualarray.txt', True))
#             os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
#             P.start()
        
	# P2 = mp.Process(target=VGG_16, args=(40, 60, 8, 'test2bit_round/test2bit2_round.txt', True))
	# P3 = mp.Process(target=VGG_16, args=(60, 80, 8, 'test2bit_round/test2bit3_round.txt', True))
	# P4 = mp.Process(target=VGG_16, args=(80, 100, 8, 'test2bit_round/test2bit4_round.txt', True))
	# P5 = mp.Process(target=VGG_16, args=(100, 120, 8, 'test2bit_round/test2bit5_round.txt', True))
	# P6 = mp.Process(target=VGG_16, args=(120, 140, 8, 'test2bit_round/test2bit6_round.txt', True))
	# P7 = mp.Process(target=VGG_16, args=(140, 160, 8, 'test2bit_round/test2bit7_round.txt', True))
	# P8 = mp.Process(target=VGG_16, args=(160, 180, 8, 'test2bit_round/test2bit8_round.txt', True))
# # 	# P9 = mp.Process(target=VGG_16, args=(100, 110, 8, 'test2bit1.txt', True))
# 	# P10 = mp.Process(target=VGG_16, args=(110, 120, 8, 'test2bit9.txt', True))
# 	# P11 = mp.Process(target=VGG_16, args=(120, 130, 8, 'test2bit10.txt', True))
# 	# P12 = mp.Process(target=VGG_16, args=(130, 140, 8, 'test2bit11.txt', True))
# 	# P13 = mp.Process(target=VGG_16, args=(140, 150, 8, 'test2bit12.txt', True))
# 	# P14 = mp.Process(target=VGG_16, args=(150, 160, 8, 'test2bit13.txt', True))
# 	# P15 = mp.Process(target=VGG_16, args=(160, 170, 8, 'test2bit14.txt', True))
# 	# P16 = mp.Process(target=VGG_16, args=(170, 180, 8, 'test2bit15.txt', True))

	# P1.start()
	# P2.start()
	# P3.start()
	# P4.start()
	# P5.start()
	# P6.start()
	# P7.start()
	# P8.start()
	# P9.start()
	# P10.start()
	# P11.start()
	# P12.start()
	# P13.start()
	# P14.start()
	# P15.start()
	# P16.start()


	# P1.join()
	#P2.join()
	#P3.join()
	#P4.join()
	#P5.join()
	#P6.join()
	#P7.join()
	#P8.join()
                           
# VGG_16(0, 25, 8, 7, 'test2bit_8.txt', True)


