import math
import numpy as np
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping
from Weight_mapping import dw_conv_mapping
import Activation_mapping
from Activation_mapping import act_fc_mapping
from Activation_mapping import act_conv_mapping
from Activation_mapping import act_dw_conv_mapping

import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set
def fc(input_act, weight_2D, weight_mapped):#input_act(ch, w_size=row_cnt*col_cnt)
    input_shape=input_act.shape
    w_shape=weight_2D.shape
    act_row_cnt=input_shape[0]
    act_ch_cnt=input_shape[1]
    #input_2D=input_act.reshape(1,)
    input_mapped=act_fc_mapping(input_act)
    #weight_mapped=fc_mapping(weight_2D)
    partial_sum=np.zeros(int(w_shape[1]))
    patch_num=1
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight_recorder=np.moveaxis(crossbar_set[int(weight_mapped[i][j])].get_array(),-1,-0) #exchange column and row
            
            
            input_vector=input_set[int(input_mapped[0][i])].get_array()
            for ch in range(cb_size):
                if cb_size*(j)+ch < w_shape[1]:
                    partial_sum[cb_size*(j)+ch]+=np.matmul(input_vector,weight_recorder[ch])
    output_act=partial_sum
    return output_act
    

def convlution(input_act, weight_4D, weight_mapped, stride): #input_act(patch_num, w_size)  weight_4D()
    w_shape=weight_4D.shape
    input_mapped=act_conv_mapping(input_act,weight_4D,stride)
    #weight_mapped=conv_mapping(weight_4D)
    patch_num=len(input_mapped)
    
    partial_sum=np.zeros((int(patch_num),int(w_shape[3])))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight_recorder=np.moveaxis(crossbar_set[int(weight_mapped[i][j])].get_array(),-1,-0) #exchange column and row
            
            for k in range(patch_num):
                
                input_vector=input_set[int(input_mapped[k][i])].get_array()
                for ch in range(cb_size):
                    if cb_size*(j)+ch < w_shape[3]:
                        partial_sum[k][cb_size*(j)+ch]+=np.matmul(input_vector,weight_recorder[ch])
    output_act=partial_sum
    output_act=np.moveaxis(output_act,-1,0)
    output_act_shape=output_act.shape
    output_ch=output_act_shape[0]
    output_row=int(math.sqrt(patch_num))
    output_act=output_act.reshape(output_act_shape[0],output_row, output_row)
    return output_act
def dw_convlution(input_act, weight_3D, weight_mapped, stride, duplicate, time_multiplex):
    input_mapped=act_dw_conv_mapping(input_act, weight_3D, stride, duplicate, time_multiplex)
    #weight_mapped=dw_conv_mapping(weight_3D, stride, duplicate, time_multiplex)
    w_shape=weight_3D.shape
    row_multiplex=math.ceil(w_shape[2]/time_multiplex)
    #print(row_multiplex)
    patch_num=int(len(input_mapped)/time_multiplex)*duplicate
    partial_sum=np.zeros((int(patch_num),int(w_shape[2])))
    #print(partial_sum.shape)
    input_shape=input_act.shape
    #print(len(input_mapped))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight_recorder=np.moveaxis(crossbar_set[int(weight_mapped[i][j])].get_array(),-1,-0) #exchange column and row
            for col in range(time_multiplex):
                #for k in range(0,patch_num,duplicate):
                #print('col')
                #print(col)
                for row in range(int(len(input_mapped)/time_multiplex)):
                    input_vector=input_set[int(input_mapped[int(col*len(input_mapped)/time_multiplex+row)][i])].get_array()
                        
                    for n in range(row_multiplex):
                        #print('n')
                        #print(n)
                        if col*row_multiplex+n<w_shape[2]:
                            for m in range(duplicate):
                                weight_col=col*duplicate*row_multiplex-j*cb_size+m+n*duplicate
                                if (cb_size*(j)+weight_col) < (w_shape[2]*duplicate) and weight_col<cb_size and weight_col>=0:                               
                                    partial_sum[int(row)*duplicate+m][col*row_multiplex+n]+=np.matmul(input_vector,weight_recorder[weight_col])
    output_act=partial_sum
    output_act=np.moveaxis(output_act,-1,0)
    output_col=int(input_shape[2]/stride)
    output_act=output_act.reshape(w_shape[2],output_col,int(patch_num/output_col))
    output_act=np.swapaxes(output_act, 1, 2)
    output_shape=output_act.shape
    for i in range(output_shape[1]-output_shape[2]):
        output_act=np.delete(output_act, -1, axis=1)
    
    return output_act
    
