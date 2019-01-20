import math
import numpy as np
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping
from Weight_mapping import dw_conv_mapping
from Weight_mapping import pw_conv_mapping
import Activation_mapping
from Activation_mapping import act_fc_mapping
from Activation_mapping import act_conv_mapping
from Activation_mapping import act_dw_conv_mapping
from Activation_mapping import act_pw_conv_mapping
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
    partial_sum=np.zeros((len(weight_mapped[0]), cb_size))
    patch_num=1
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() #exchange column and row
            
            
            input_vector=input_set[int(input_mapped[0][i])].get_array()
            #range_ch=min(cb_size, w_shape[1]-cb_size*j)
            #for ch in range(range_ch):
                #if cb_size*(j)+ch < w_shape[1]:
            partial_sum[j]+=np.matmul(input_vector,weight)
    output_act=partial_sum.flatten()
    output_act=output_act[:w_shape[1]]
    return output_act
    

def convlution(input_act, weight_4D, weight_mapped, stride): #input_act(patch_num, w_size)  weight_4D()
    w_shape=weight_4D.shape
    input_mapped=act_conv_mapping(input_act,weight_4D,stride)
    #weight_mapped=conv_mapping(weight_4D)
    patch_num=int(len(input_mapped))
    
    partial_sum=np.zeros((patch_num,len(weight_mapped[0]),cb_size))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() #exchange column and row
            
            for k in range(patch_num):
                
                input_vector=input_set[int(input_mapped[k][i])].get_array()
                #range_ch=min(cb_size, w_shape[3]-cb_size*j)
                #for ch in range(range_ch):
                    #if cb_size*(j)+ch < w_shape[3]:
                partial_sum[k][j]+=np.matmul(input_vector,weight)
    output_act=partial_sum.reshape((patch_num,len(weight_mapped[0])*cb_size))
    
    output_act=np.moveaxis(output_act,-1,0)
    output_act=output_act[:w_shape[3]]
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
    patch_num=int(len(input_mapped)/time_multiplex)
    partial_sum=np.zeros((patch_num,len(weight_mapped[0]),cb_size))
    #print(partial_sum.shape)
    input_shape=input_act.shape
    #print(len(input_mapped))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() 
            #for col in range(time_multiplex):
                #for k in range(0,patch_num,duplicate):
                #print('col')
                #print(col)
            col=int(j*cb_size/(row_multiplex*duplicate))
            for row in range(patch_num):
                
                input_vector=input_set[int(input_mapped[int(col*patch_num+row)][i])].get_array()
               
                if int((j+1)*cb_size/(row_multiplex*duplicate))>col and col+1<time_multiplex:
                    partial_sum[row][j][0:(col+1)*row_multiplex*duplicate-j*cb_size]+=np.matmul(input_vector, weight[:,:(col+1)*row_multiplex*duplicate-j*cb_size])
                    input_vector_1=input_set[int(input_mapped[int((col+1)*patch_num+row)][i])].get_array()
                    partial_sum[row][j][(col+1)*row_multiplex*duplicate-j*cb_size:cb_size]+=np.matmul(input_vector_1, weight[:,(col+1)*row_multiplex*duplicate-j*cb_size:cb_size])
                    
                else:
                    partial_sum[row][j]+=np.matmul(input_vector, weight)
                #if int((j-1)*cb_size/(row_multiplex*duplicate))<col and j-1>=0:
                    #weight = crossbar_set[int(weight_mapped[i][j-1])].get_array()
                    #partial_sum[row][j-1][col*row_multiplex*duplicate-(j-1)*cb_size :cb_size]+=np.matmul(input_vector, weight[:,col*row_multiplex*duplicate-(j-1)*cb_size :cb_size])
 

                    #if j*cb_size<(col+1)*row_multiplex*duplicate and (j+1)*cb_size>=(col+1)*row_multiplex*duplicate and col+1<time_multiplex and j+1<len(weight_mapped[0]):

                     #   input_vector=input_set[int(input_mapped[int(col*patch_num+row)][i])].get_array()
                      #  partial_sum[col*patch_num+row][j][0:(col+1)*row_multiplex*duplicate-j*cb_size]+=np.matmul(input_vector, weight[:,:(col+1)*row_multiplex*duplicate-j*cb_size])
                        #print('if', (col+1)*row_multiplex*duplicate-j*cb_size, row)
                        #print(partial_sum[0][0])
                    #elif (j-1)*cb_size<col*row_multiplex*duplicate and j*cb_size>=col*row_multiplex*duplicate and (j-1)>=0:
                        #partial_sum[col*patch_num+row][j]+=np.matmul(input_vector, weight)
                     #   weight = crossbar_set[int(weight_mapped[i][j-1])].get_array()
                        #print(col*patch_num*duplicate-(j-1)*cb_size)
                      #  partial_sum[col*patch_num+row][j-1][col*row_multiplex*duplicate-(j-1)*cb_size :cb_size]+=np.matmul(input_vector, weight[:,col*row_multiplex*duplicate-(j-1)*cb_size :cb_size])
                       # print('elif',col*row_multiplex*duplicate-(j-1)*cb_size, j-1)
                       # print(partial_sum[-1][j-1])
                    #else:
                     #   partial_sum[col*patch_num+row][j]+=np.matmul(input_vector, weight)

    #print(partial_sum)
    #print(partial_sum.shape)
    partial_sum=partial_sum.reshape((patch_num, int(len(weight_mapped[0])*cb_size)))
    #partial_sum=np.moveaxis(partial_sum, -1, 0)
    partial_sum=partial_sum[:,:row_multiplex*duplicate*time_multiplex]
    #partial_sum=np.moveaxis(partial_sum, 1, 0)
    partial_sum=partial_sum.reshape((patch_num, time_multiplex, row_multiplex, duplicate))
    #partial_sum=np.moveaxis(partial_sum,-1,-2)
    partial_sum=np.moveaxis(partial_sum,-1,1)
    #partial_sum=np.swapaxes(partial_sum,0,-1)
    partial_sum=partial_sum.reshape((patch_num*duplicate, row_multiplex*time_multiplex))
    partial_sum=np.swapaxes(partial_sum,0,-1)
    partial_sum=partial_sum[:w_shape[2]]
    #partial_sum=np.moveaxis(partial_sum, -1, 0)               
                    #index= w_shape[2]-col*row_multiplex
                    #range_n=min(row_multiplex, index)
                    #for n in range(range_n):
                        #print('n')
                        #print(n)
                        #if col*row_multiplex+n<w_shape[2]:
                     #   range_m=min(duplicate, (index-n)*duplicate)
                        
                        #for m in range(range_m):
                      #  weight_col=col*duplicate*row_multiplex-j*cb_size+n*duplicate
                        #range_m=
                        #(cb_size*(j)+weight_col) < (w_shape[2]*duplicate) and
                       # if weight_col+range_m<cb_size and weight_col>=0:                               
                        #    partial_sum[col*row_multiplex+n][int(row)*duplicate:int(row)*duplicate+range_m]+=np.matmul(weight_recorder[weight_col:weight_col+range_m],input_vector)
    output_act=partial_sum
    #output_act=np.moveaxis(output_act,-1,0)
    output_col=int(input_shape[2]/stride)
    output_act=output_act.reshape(w_shape[2],output_col,int(patch_num*duplicate/output_col))
    output_act=np.swapaxes(output_act, 1, 2)
    output_shape=output_act.shape
    for i in range(output_shape[1]-output_shape[2]):
        output_act=np.delete(output_act, -1, axis=1)
    
    return output_act
def pw_convlution(input_act, weight_4D, weight_mapped, stride): #input_act(patch_num, w_size)  weight_4D()
    w_shape=weight_4D.shape
    input_mapped=act_pw_conv_mapping(input_act,weight_4D,stride)
    #weight_mapped=conv_mapping(weight_4D)
    patch_num=int(len(input_mapped))
    
    partial_sum=np.zeros((patch_num,len(weight_mapped[0]),cb_size))
    for j in range(len(weight_mapped[0])):#number of arrays per row
        for i in range(len(weight_mapped)):#number of arrays per column
            weight = crossbar_set[int(weight_mapped[i][j])].get_array() #exchange column and row
            
            for k in range(patch_num):
                #for t in range(len(input_mapped[0])):#number of arrays per row
                input_vector=input_set[int(input_mapped[k][i])].get_array()
                #range_ch=min(cb_size, w_shape[3]-cb_size*j)
                #for ch in range(range_ch):
                    #if cb_size*(j)+ch < w_shape[3]:
                partial_sum[k][j]+=np.matmul(input_vector,weight)
    output_act=partial_sum.reshape((patch_num,len(weight_mapped[0])*cb_size))
    
    output_act=np.moveaxis(output_act,-1,0)
    output_act=output_act[:w_shape[3]]
    output_act_shape=output_act.shape
    output_ch=output_act_shape[0]
    output_row=int(math.sqrt(patch_num))
    output_act=output_act.reshape(output_act_shape[0],output_row, output_row)
    return output_act