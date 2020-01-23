import math
import numpy as np
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping
import Activation_mapping
from Activation_mapping import act_fc_mapping, act_fc_mapping_int
from Activation_mapping import act_conv_mapping, act_conv_mapping_int
import gl
from gl import cb_size, data
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set, scale2bit
import matplotlib.pyplot as plt
from gl import latency

def max_pool(input_act, pool_window):
    R = int(input_act.shape[1]/pool_window)
    C = int(input_act.shape[2]/pool_window)
    output_act = np.zeros((input_act.shape[0], R, C))
    for n in range(output_act.shape[0]):
        output_act[n, :, :] = input_act[n, :input_act.shape[1], :input_act.shape[2]].reshape(R, pool_window, C, pool_window).max(axis=(1, 3))
    return output_act

def fc(input_act, weight_2D, weight_mapped):#input_act(ch, w_size=row_cnt*col_cnt)
    input_shape = input_act.shape
    w_shape = weight_2D.shape
    w_shape = w_shape[2:4]
    act_row_cnt = input_shape[0]
    act_ch_cnt = input_shape[1]
    #input_2D=input_act.reshape(1,)
    input_mapped=act_fc_mapping(input_act)
    # print('fc_shape=', input_mapped.shape)
   
    partial_sum = np.zeros((len(weight_mapped[0]), cb_size[1]))
    patch_num = 1
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() #exchange column and row
            
            
            input_vector = input_set[int(input_mapped[0][i])].get_array()
            #range_ch = min(cb_size, w_shape[1]-cb_size*j)
            #for ch in range(range_ch):
                #if cb_size*(j)+ch < w_shape[1]:
            partial_sum[j] += np.matmul(input_vector,weight)
    output_act = partial_sum.flatten()
    output_act = output_act[:w_shape[1]]
    return output_act

def fc_int(input_act, scale_input, weight_2D, weight_mapped, scale_weight, bias, scale_bias, num_bits, if_partial_ADC):
    input_shape = input_act.shape
    w_shape = weight_2D.shape
    w_shape = w_shape[2:4]
    act_row_cnt = input_shape[0]
    act_ch_cnt = input_shape[1]
    #input_2D = input_act.reshape(1,)
    input_mapped = act_fc_mapping(input_act)
    # print('fc_shape=', input_mapped.shape)
   
    partial_sum = np.zeros((len(weight_mapped[0]), cb_size[1]))
    patch_num = 1
    latency.append(1)

    #takes larger value between absolute values of the max and min of input_act, weight_2D, and bias
    input_scale = max(abs(np.max(input_act)), abs(np.min(input_act)))
    weight_scale = max(abs(np.max(weight_2D)), abs(np.min(weight_2D)))
    bias_scale = max(abs(np.max(bias)), abs(np.min(bias)))

    
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])]#.get_array() #exchange column and row
            
            
            input_vector = input_set[int(input_mapped[i])].get_array()
            #range_ch=min(cb_size, w_shape[1]-cb_size*j)
            #for ch in range(range_ch):
                #if cb_size*(j)+ch < w_shape[1]:
            # if if_partial_ADC:
            partial_sum[j] += np.squeeze(weight.accumulate(input_vector, weight.get_array(), input_scale, num_bits, if_partial_ADC, False))
            #     partial_sum[j]+=(np.matmul(input_vector,weight) * 127 / (2**31-1)).astype('int')
            # else:
            #     partial_sum[j]+=np.matmul(input_vector,weight)
    output_act = partial_sum.flatten()
    output_act = output_act[:w_shape[1]]


    int_acc = output_act
    S1 = scale_input
    S2 = scale_weight
    bias  = np.round(bias * (2**(num_bits)))
    # if if_partial_ADC:
    #     int_acc = (int_acc * cb_size * input_scale * weight_scale / (2**12-1))
        # bias = np.round(np.round(bias * 127 / bias_scale) * bias_scale / 127)
    act_float = S1*S2*(int_acc + bias) / (2**(num_bits))
    return act_float


def Conv2D_base(input_mapped, w_shape, weight_mapped, stride, if_partial_ADC):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    patch_num = int(len(input_mapped))

    partial_sum = np.zeros((patch_num,len(weight_mapped[0]),cb_size[1]))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() #exchange column and row
            
            for k in range(patch_num):
                input_vector=input_set[int(input_mapped[k][i])].get_array()
                #range_ch=min(cb_size, w_shape[3]-cb_size*j)
                #for ch in range(range_ch):
                    #if cb_size*(j)+ch < w_shape[3]:
                partial_sum[k][j]+=np.matmul(input_vector,weight)
    output_act = partial_sum.reshape((patch_num,len(weight_mapped[0])*cb_size[1]))
    
    output_act = np.moveaxis(output_act,-1,0)
    output_act = output_act[:w_shape[3]]
    output_act_shape=output_act.shape
    output_ch = output_act_shape[0]
    output_row = int(math.sqrt(patch_num))
    output_act = output_act.reshape(output_act_shape[0],output_row, output_row)
    return output_act

def Conv2D(input_act, weight_4D, weight_mapped, stride, if_partial_ADC): #input_act(patch_num, w_size)  weight_4D()
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    w_shape = weight_4D.shape
    input_mapped = act_conv_mapping(input_act,weight_4D,stride)[0]
    return Conv2D_base(input_mapped, w_shape, weight_mapped, stride, if_partial_ADC)

def Conv2D_quant_uint(input_act, scale_input, offset_input, weight_4D, weight_mapped, scale_weight, offset_weight, bias, scale_bias, stride, if_partial_ADC):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    w_shape = weight_4D.shape
    input_mapped = act_conv_mapping(input_act, weight_4D, stride)
    uint_acc = Conv2D_base(input_mapped, w_shape, weight_mapped, stride, if_partial_ADC)
    
    i = input_act.shape[0]
    j = input_act.shape[1]
    k = weight_4D.shape[1]
    a1_array = np.tile(input_act.sum(axis=1), (k, 1)).T
    a2_array = np.tile(weight_4D.sum(axis=0),(i, 1))
    bias_array = np.tile(bias, (i, 1))
    N = j
    Z1 = offset_input
    Z2 = offset_weight
    S1 = scale_input
    S2 = scale_weight
    act_int_float = S1*S2*(N*Z1*Z2 - Z1*a2_array - Z2*a1_array + uint_acc + bias_array)
    return act_int_float

def Conv2D_quant_int(input_act, scale_input, weight_4D, weight_mapped, scale_weight, bias, scale_bias, stride, num_bits, if_partial_ADC):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    w_shape = weight_4D.shape
    input_mapped = act_conv_mapping(input_act, weight_4D, stride)
    # print('conv_shape=', input_mapped.shape)
    patch_num = int(len(input_set[int(input_mapped[0])].get_array()))
    latency.append(patch_num)
    partial_sum = np.zeros((patch_num,len(weight_mapped[0]),cb_size[1]))

    #takes larger value between absolute values of the max and min of input_act, weight_4D, and bias
    input_scale = max(abs(np.max(input_act)), abs(np.min(input_act)))
    weight_scale = max(abs(np.max(weight_4D)), abs(np.min(weight_4D)))
    bias_scale = max(abs(np.max(bias)), abs(np.min(bias)))

    output_row = int(math.sqrt(patch_num))
    output_ch = len(weight_mapped[0])*cb_size[1]
    if input_scale == 0:
        output_act = np.zeros((output_ch, output_row, output_row))
    if input_scale > 0:
        for j in range(len(weight_mapped[0])):#number of arrays per row
            for i in range(len(weight_mapped)):#number of arrays per column
                weight = crossbar_set[int(weight_mapped[i][j])]#.get_array() #exchange column and row
                input_vector=input_set[int(input_mapped[i])].get_array()
                #range_ch=min(cb_size, w_shape[3]-cb_size*j)
                #for ch in range(range_ch):
                    #if cb_size*(j)+ch < w_shape[3]:
                # if if_partial_ADC:
                #     partial_sum[k][j]+=(np.matmul(input_vector,weight)*127/(2**31-1)).astype('int')
                # else:
                #     partial_sum[k][j]+=np.matmul(input_vector,weight)
                partial_sum[:,j,:] +=  weight.accumulate(input_vector, weight.get_array(), input_scale, num_bits, if_partial_ADC, False)
        output_act = partial_sum.reshape((patch_num,len(weight_mapped[0])*cb_size[1]))
        
        output_act = np.moveaxis(output_act,-1,0)
        output_act = output_act[:w_shape[3]]
        output_act = output_act.reshape(w_shape[3],output_row, output_row)
    
    int_acc = output_act
    i = int_acc.shape[0]
    j = int_acc.shape[1]
    k = weight_4D.shape[1]
    S1 = scale_input
    S2 = scale_weight
    if input_scale > 0:
        bias = np.round(bias * (2**(num_bits)))
    # act_float = S2*(int_acc + bias[:, np.newaxis, np.newaxis])
    # if if_partial_ADC:
    #     int_acc = (int_acc * cb_size * input_scale * weight_scale / (2**12-1))
        # bias = np.round(np.round(bias * 127 / bias_scale) * bias_scale / 127)
        act_float = S1*S2*(int_acc + bias[:, np.newaxis, np.newaxis]) / (2**(num_bits))
        return act_float
    else:
        return output_act


