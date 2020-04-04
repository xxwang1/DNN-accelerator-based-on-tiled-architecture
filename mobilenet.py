import numpy as np
import math
import Layer_function
from Layer_function import fc, fc_int
from Layer_function import Conv2D, Conv2D_quant_uint, Conv2D_quant_int
from Layer_function import dw_convlution, dw_convlution_int
from Layer_function import pw_convlution, pw_convlution_int
import nonlinear
from nonlinear import batchnorm
from nonlinear import ReLu, Relu6_quant, Relu8_quant
import img_TFRecords
from img_TFRecords import image_to_act_mobilenet

from weight_loading import load_from_tflite, load_from_pb
import Weight_mapping
from Weight_mapping import fc_mapping, fc_mapping_int
from Weight_mapping import conv_mapping, conv_mapping_int
from Weight_mapping import dw_conv_mapping, dw_conv_mapping_int
from Weight_mapping import pw_conv_mapping, pw_conv_mapping_int
import gl
from gl import cb_size, data, scale_mode, energy
from gl import (w_tot, w_int)
from gl import crossbar_set, latency
from gl import input_set, scale2bit, array2bit, mapping_mode
from pylab import *
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
import json


def MobileNet_int(start_img, end_img, ADC_bit, Pi, if_partial_ADC):
    # weights_quant = load_from_tflite('int')
    weights_quant_tflite = load_from_tflite('int')
    weights_quant = weights_quant_tflite
    Relu6_scale = 5.999761581420898/255
    weight_scale = []
    weight_scale.append(0)
    cb_count = []
    weight_sum = []
    #layers
    #Conv2d_0 input(ch, row, col)
    #weights mapping
    ##############################################################################
    stride_depthwise=np.array([1,2,1,2,1,2,1,1,1,1,1,2,1])
    stride_pointwise=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
    crossbar_cnt = 0
    weight_Conv2d_0_mapped = conv_mapping(weights_quant['Conv2d_0/weights']['array'], mode=mapping_mode)
    # weight_sum.append(np.sum(np.abs(weights_quant['Conv2d_0/weights']['array'])))
    # cb_count.append(weight_Conv2d_0_mapped.shape[0]*weight_Conv2d_0_mapped.shape[1])
    weight_sum_layer = 0
    for j in range(len(weight_Conv2d_0_mapped[0])):#number of arrays per row
        for i in range(len(weight_Conv2d_0_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_Conv2d_0_mapped[i][j])]#.get_array() 
    #         weight_sum_layer += np.sum(weight.get_array())
    # weight_sum.append(weight_sum_layer)
            for m in range((weight.weight_scale).shape[0]):
    # #             if (weight.weight_scale)[m] > weight_scale[-1]:
    # #                 weight_scale[-1] = (weight.weight_scale)[m]
                if (weight.weight_scale)[m] !=0:#>=655.3: 
                    weight_scale.append(array2bit(weight.weight_scale)[m])
    #             if (weight.weight_scale)[1][m] !=0:
    #                 weight_scale.append(array2bit(weight.weight_scale[1])[m])
    # # # crossbar_cnt += weight_Conv2d_0_mapped.shape[0] * weight_Conv2d_0_mapped.shape[1]
    # # print('conv_0', weight_Conv2d_0_mapped.shape[0] * weight_Conv2d_0_mapped.shape[1])
    # if scale_mode == 'per_layer':
    #     for j in range(len(weight_Conv2d_0_mapped[0])):#number of arrays per row
    #         for i in range(len(weight_Conv2d_0_mapped)):#number of arrays per column
    #             crossbar_set[int(weight_Conv2d_0_mapped[i][j])].weight_scale = np.ones(cb_size[1]) * weight_scale[-1] 
    for i in range(1,14):
        # weight_scale.append(0)
        locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'] = dw_conv_mapping(weights_quant['Conv2d_'+str(i)+'_depthwise/weights']['array'], stride_depthwise[i-1], 2, 2, 
                                                                                mode=mapping_mode)
        # weight_sum.append(np.sum(np.abs(weights_quant['Conv2d_'+str(i)+'_depthwise/weights']['array'])))
        weight_sum_layer = 0
        for j in range(len(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'][0])):#number of arrays per row
            for k in range(len(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'])):#number of arrays per column
                weight = crossbar_set[int(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'][k][j])]#.get_array() 
        #         weight_sum_layer += np.sum(weight.get_array())
        # weight_sum.append(weight_sum_layer)
                for m in range((weight.weight_scale).shape[0]):
        # #             if (weight.weight_scale)[m] > weight_scale[-1]:
        # #                 weight_scale[-1] = (weight.weight_scale)[m]
                    if (weight.weight_scale)[m] !=0:#>=655.3:
                        weight_scale.append(array2bit(weight.weight_scale)[m])
        #             if (weight.weight_scale)[1][m] !=0:
        #                 weight_scale.append(array2bit(weight.weight_scale[1])[m])
        cb_count.append(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'].shape[0] * locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'].shape[1])
        # # print('dw_i', locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'].shape[0] * locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'].shape[1])
        # if scale_mode == 'per_layer':
        #     for j in range(len(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'][0])):#number of arrays per row
        #         for k in range(len(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'])):#number of arrays per column
        #             crossbar_set[int(locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'][k][j])].weight_scale = np.ones(cb_size[1]) * weight_scale[-1] 
        # weight_scale.append(0)
        locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'] = conv_mapping(weights_quant['Conv2d_'+str(i)+'_pointwise/weights']['array'], 
                                                                             mode=mapping_mode)
        # weight_sum.append(np.sum(np.abs(weights_quant['Conv2d_'+str(i)+'_pointwise/weights']['array'])))
        weight_sum_layer = 0
        for j in range(len(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'][0])):#number of arrays per row
            for k in range(len(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'])):#number of arrays per column
                weight = crossbar_set[int(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'][k][j])]#.get_array()
        #         weight_sum_layer += np.sum(weight.get_array())
        # weight_sum.append(weight_sum_layer) 
                for m in range((weight.weight_scale).shape[0]):
        # #             if (weight.weight_scale)[m] > weight_scale[-1]:
        # #                 weight_scale[-1] = (weight.weight_scale)[m]
                    if (weight.weight_scale)[m] !=0:#>=655.3:
                        weight_scale.append(array2bit(weight.weight_scale)[m])
        #             if (weight.weight_scale)[1][m] !=0:
        #                 weight_scale.append(array2bit(weight.weight_scale[1])[m])
        cb_count.append(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'].shape[0] * locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'].shape[1])
        # # print('pw_i', locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'].shape[0] * locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'].shape[1])
        # if scale_mode == 'per_layer':
        #     for j in range(len(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'][0])):#number of arrays per row
        #         for k in range(len(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'])):#number of arrays per column
        #             crossbar_set[int(locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'][k][j])].weight_scale = np.ones(cb_size[1]) * weight_scale[-1]
    # weight_scale.append(0)
    weight_fc_mapped=fc_mapping(weights_quant['FC/weights']['array'], 
                                mode=mapping_mode)
    # weight_sum.append(np.sum(np.abs(weights_quant['FC/weights']['array'])))
    weight_sum_layer = 0
    for j in range(len(weight_fc_mapped[0])):#number of arrays per row
        for k in range(len(weight_fc_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_fc_mapped[k][j])]#.get_array() 
    #         weight_sum_layer += np.sum(weight.get_array())
    # weight_sum.append(weight_sum_layer)
            for m in range((weight.weight_scale).shape[0]):
    # #             if (weight.weight_scale)[m] > weight_scale[-1]:
    # #                 weight_scale[-1] = (weight.weight_scale)[m]
                if (weight.weight_scale)[m] !=0:#>=655.3:
                        weight_scale.append(array2bit(weight.weight_scale)[m])
    #             if (weight.weight_scale)[1][m] !=0:
    #                 weight_scale.append(array2bit(weight.weight_scale[1])[m])
    # print('weight_sum=', weight_sum)
    cb_count.append(weight_fc_mapped.shape[0] * weight_fc_mapped.shape[1])
    # weight_scale_array = np.histogram(np.array(weight_scale), bins=(range(0, 2250, 25)))
    # np.savetxt('mb_8scale.txt', weight_scale_array)
    # with open('mb_weight_sum.txt', 'w') as f:
    #     for item in weight_sum:
    #         f.write("%s\n" % item)
    # # print('fc', weight_fc_mapped.shape[0] * weight_fc_mapped.shape[1])
    # # print('crossbar number=', crossbar_cnt)
    # # print('weight_scale=', weight_scale)
    # if scale_mode == 'per_layer':
    #     for j in range(len(weight_fc_mapped[0])):#number of arrays per row
    #         for k in range(len(weight_fc_mapped)):#number of arrays per column
    #             crossbar_set[int(weight_fc_mapped[k][j])].weight_scale = np.ones(cb_size[1]) * weight_scale[-1]
    matplotlib.rcParams.update({'font.size': 16})
    plt.hist(np.array(weight_scale), bins=100)
    plt.xlabel('Partial sum scale', {'size':20})
    plt.ylabel('Count', {'size':20})
    ax = plt.gca()
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(xfmt)
    plt.tick_params(labelsize=16)
    plt.show()

    #read images
    ##############################################################################
    path ="../ILSVRC2012_img_val/"    #指定需要读取文件的目录
    files =os.listdir(path) #采用listdir来读取所有文件
    files.sort() #排序
    #def MobileNet(start_img, end_img):
    for i in range(start_img,end_img):
        energy.append(0)
        file_=files[i]
        if not os.path.isdir(path +file_):#判断该文件是否是一个文件夹
            f_name = str(file_)
            print(f_name)
            input_act_pos=image_to_act_mobilenet(f_name)
            input_act_neg = image_to_act_mobilenet(f_name) * (-1)
            input_pos = ReLu(np.round(input_act_pos*128))
            input_neg = ReLu(np.round(input_act_neg*128))
            Relu8_scale = 8/256
            input_act=image_to_act_mobilenet(f_name)
            output_Conv2d_0_p = Conv2D_quant_int(input_pos,
                                                scale2bit(weights_quant['input']['quantization'][0]),
                                                weights_quant['Conv2d_0/weights']['array'],
                                                weight_Conv2d_0_mapped,
                                                (weights_quant['Conv2d_0/weights']['quantization'][0]),
                                                weights_quant['Conv2d_0/bias']['array'],
                                                scale2bit(weights_quant['Conv2d_0/bias']['quantization'][0]),
                                                2, ADC_bit, if_partial_ADC)
            output_Conv2d_0_n = (-1) * Conv2D_quant_int(input_neg,
                                                scale2bit(weights_quant['input']['quantization'][0]),
                                                weights_quant['Conv2d_0/weights']['array'],
                                                weight_Conv2d_0_mapped,
                                                (weights_quant['Conv2d_0/weights']['quantization'][0]),
                                                np.zeros_like(weights_quant['Conv2d_0/bias']['array']),
                                                scale2bit(weights_quant['Conv2d_0/bias']['quantization'][0]),
                                                2, ADC_bit, if_partial_ADC)
            output_Conv2d_0 = output_Conv2d_0_p + output_Conv2d_0_n
            
            input_Conv2d_1_depthwise = Relu6_quant(output_Conv2d_0, scale2bit(Relu6_scale))
            # with open('data.json', 'w') as fp:
            #     json.dump(data, fp)
            #     fp.close()

            for j in range(1,14):
                locals()['output_Conv2d_'+str(j)+'_depthwise'] = dw_convlution_int(locals()['input_Conv2d_'+str(j)+'_depthwise'],
                                                                                #    Relu8_scale,
                                                                                   scale2bit(weights_quant['Conv2d_0/Relu6']['quantization'][0]),
                                                                                   weights_quant['Conv2d_'+str(j)+'_depthwise/weights']['array'],
                                                                                   locals()['weight_Conv2d_'+str(j)+'_depthwise_mapped'],
                                                                                   (weights_quant['Conv2d_'+str(j)+'_depthwise/weights']['quantization'][0]),
                                                                                   weights_quant['Conv2d_'+str(j)+'_depthwise/bias']['array'],
                                                                                   stride_depthwise[j-1], 2, 2, ADC_bit, if_partial_ADC)
                
                locals()['input_Conv2d_'+str(j)+'_pointwise'] = Relu6_quant(locals()['output_Conv2d_'+str(j)+'_depthwise'], scale2bit(Relu6_scale))#(weights_quant['Conv2d_0/Relu6']['quantization'][0]))

                locals()['output_Conv2d_'+str(j)+'_pointwise'] = Conv2D_quant_int(locals()['input_Conv2d_'+str(j)+'_pointwise'], 
                                                                                #    Relu8_scale,
                                                                                   scale2bit(weights_quant['Conv2d_0/Relu6']['quantization'][0]),
                                                                                   weights_quant['Conv2d_'+str(j)+'_pointwise/weights']['array'],
                                                                                   locals()['weight_Conv2d_'+str(j)+'_pointwise_mapped'],
                                                                                   (weights_quant['Conv2d_'+str(j)+'_pointwise/weights']['quantization'][0]),
                                                                                   weights_quant['Conv2d_'+str(j)+'_pointwise/bias']['array'],
                                                                                   scale2bit(weights_quant['Conv2d_'+str(j)+'_pointwise/bias']['quantization'][0]),
                                                                                   stride_pointwise[j-1], ADC_bit, if_partial_ADC)
                
                locals()['input_Conv2d_'+str(j+1)+'_depthwise'] = Relu6_quant(locals()['output_Conv2d_'+str(j)+'_pointwise'], scale2bit(Relu6_scale)) #(weights_quant['Conv2d_0/Relu6']['quantization'][0]))
                #print(locals()['input_Conv2d_'+str(i+1)+'_depthwise'].shape)

            #print(locals()['input_Conv2d_14_depthwise'])
            
            #Average pooling
            input_fc=np.mean(locals()['input_Conv2d_14_depthwise'],axis=1)
            input_fc=np.mean(input_fc,axis=1)
            # print(input_fc.shape)
            
            input_fc_shape=input_fc.shape
            input_fc=input_fc.reshape(1,input_fc_shape[0])
            
            # output_fc=fc(input_fc,weight_fc,weight_fc_mapped)
            logits = fc_int(input_fc, 
                            #    Relu8_scale,
                               scale2bit(weights_quant['Conv2d_0/Relu6']['quantization'][0]),
                               weights_quant['FC/weights']['array'],
                               weight_fc_mapped,
                               (weights_quant['FC/weights']['quantization'][0]),
                               weights_quant['FC/bias']['array'],
                               ADC_bit, if_partial_ADC)
            

            output_fc = logits

            # print(output_fc.shape)
            f=open(Pi,'a+')
            f.write(f_name + ' ') #write to result file
            # print(np.max(output_fc), np.argmax(output_fc),file=f)
            top5 = np.argsort(output_fc)[-5:][::-1]
            print(np.max(output_fc), top5,file=f)
            f.close()

            del input_set[:]
            ground_truth = []
            for ground in open('imagenet2012_val_ground_truth.txt'):
                ground_truth.append(ground[7:-3])
            print('\n'+ 'Classification result of '+ f_name + ' is:')
            print('1. '+str(top5[0])+' '+ground_truth[top5[0]-1])
            print('2. '+str(top5[1])+' '+ground_truth[top5[1]-1])
            print('3. '+str(top5[2])+' '+ground_truth[top5[2]-1])
            print('4. '+str(top5[3])+' '+ground_truth[top5[3]-1])
            print('5. '+str(top5[4])+' '+ground_truth[top5[4]-1])
            # with open('mb_latency.txt', 'w') as f:
            #     for item in latency:
            #         f.write("%s\n" % item)
            # del result
    # data['input_activation'] = np.array(data['input_activation'])
    # data['weight_array'] = np.array(data['weight_array'])
    # data['output_activation'] = np.array(data['output_activation'])
    



MobileNet_int(12541, 12542, 8, 'mb_exp_dualarray_100ratio.txt', True)
# MobileNet_int(49, 50, 7, '30.txt', True)
# print('energy=', energy)
# print('average energy=', mean(energy))