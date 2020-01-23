# Forward Information
import tensorflow as tf
import math
import numpy as np
import Activation
from Activation import Activation
import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set
from gl import mapping_mode

#creates a 10x10x4 array of values 0-399
input_act = np.arange(400)  #28*28
input_act = input_act.reshape(10,10,4)
input_act = np.moveaxis(input_act, -1, 0)   #move axes of array to new positions - new shape: (4,10,10)

# Forward Information
import math
#creates a 10x10x4 array of values 0-399
input_act = np.arange(400)  #28*28
input_act = input_act.reshape(10,10,4)
input_act = np.moveaxis(input_act, -1, 0)   #move axes of array to new positions - new shape: (4,10,10)


def zero_padding(input_act, weight_row, weight_col, stride, duplicate):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    input_extend = input_act
    w_extend_row = weight_row+stride*(duplicate-1)
    stride_extend = stride*duplicate
    input_shape = input_act.shape
    
    #left_row=(input_shape[1]-w_extend_row)%(stride_extend)
    #left_col=(input_shape[2]-weight_col)%stride
    #if left_row!=0:
    extend_row = int((math.ceil(input_shape[1]/stride)-1)*stride+weight_row-input_shape[1])
    duplicate_num = math.ceil(input_shape[1]/stride/duplicate)
    extend_row_duplicate = int((duplicate_num-1)*stride_extend+w_extend_row-input_shape[1])
    #else:
    #extend_row = 0
    pad_row_0 = int(math.floor(extend_row/2))
    pad_row_1 = int(extend_row_duplicate-pad_row_0)
    #if left_col != 0:
    extend_col = int((math.ceil(input_shape[2]/stride)-1)*stride+weight_col-input_shape[2])
    #else:
    #    extend_col = 0
    pad_col_0 = int(math.floor(extend_col/2))
    pad_col_1 = int(extend_col-pad_col_0)
    input_extend = np.pad(input_act, [(0,0), (pad_row_0, pad_row_1), (pad_col_0, pad_col_1)], 'constant')
    #print(input_extend)
    #print(input_extend.shape)
    return input_extend

def act_fc_mapping(input_2D, mode=mapping_mode): # Get a 2D activation return a 2D input vector map
    #input_2D(patch_num, w_size)

    if mode == 'neg_act':
        #creates 2 separate arrays
        #input_pos is identical to input_2D
        #input_neg has values of input_2D except with signs flipped
        input_pos = input_2D
        input_neg = input_2D * (-1)


        input_to_map = np.zeros((input_2D.shape[0], input_2D.shape[1]*2))
        input_to_map[:,::2] = input_pos
        input_to_map[:,1::2] = input_neg
        return act_fc_mapping_base(input_to_map)
    elif mode == None or mode == 'dual_array' or mode == 'ideal':
        return np.hstack((act_fc_mapping_base(input_2D), act_fc_mapping_base(input_2D*(-1))))
def act_fc_mapping_base(input_to_map):
    input_shape = input_to_map.shape
    patch_num = input_shape[0]
    cb_array_col_cnt = math.ceil(input_shape[1]/cb_size[0]) #number of mapped arrays
    
    input_mapped = np.empty([cb_array_col_cnt])            
    for cb_col in range(cb_array_col_cnt):
        input_set.append(Activation(patch_num=patch_num))
        # Set the range of the array slice by defining starting and ending col
        col_strt = cb_col*cb_size[0]
        if cb_col == cb_array_col_cnt-1:    # execute if last col
            col_end = -1
        else:
            col_end = (cb_col+1)*cb_size[0]
        
        # Create new array object and update the array stored with sliced activation array
        if col_end == -1:   #if last col
            input_set[-1].update_array(input_to_map[:, col_strt:])
        else:
            input_set[-1].update_array(input_to_map[:, col_strt:(col_end)])
        
        new_cb_id = len(input_set)-1
        input_mapped[cb_col] = new_cb_id
    return input_mapped

def act_conv_mapping(input_act, weight_4D, stride):
    #input_act(ch, row, col)  weight_4D(row, col, ch, NF)
    input_shape = input_act.shape
    input_row_cnt = input_shape[1]
    input_col_cnt = input_shape[2]
    input_ch_cnt = input_shape[0]
    input_list = []
    input_list_vec = []
    w_shape = weight_4D.shape
    
    input_extend = zero_padding(input_act,w_shape[0],w_shape[1],stride,1)

    input_extend_shape = input_extend.shape
    input_extend_row_cnt = input_extend_shape[1]
    input_extend_col_cnt = input_extend_shape[2]
    w_size = (w_shape[0]*w_shape[1]*w_shape[2])
        
    for input_row in range(0,input_extend_row_cnt-w_shape[0]+1,stride):
        for input_col in range(0,input_extend_col_cnt-w_shape[1]+1,stride): 
            conv_window = np.moveaxis(input_extend[:, input_row : input_row+w_shape[0], input_col : input_col+w_shape[1]], 0, -1).flatten()
            input_list_vec.append(conv_window)
            # for j in range(w_shape[0]):
            #     for i in range(w_shape[1]):
            #         for input_ch in range(input_ch_cnt):
            #             input_list.append(input_extend[input_ch][input_row+j][input_col+i])
                    # input_list.append(input_extend[0:, input_row+j, input_col+i])
    # input_array=np.array(input_list) #1D, w*size*patch_num
    input_array = np.array(input_list_vec).flatten()
    # input_array =input_array.flatten('F')
    input_size = len(input_array)
    patch_num = int(input_size/w_size)
    input_2D = input_array.reshape(patch_num, w_size)
    
    return act_fc_mapping(input_2D)



def act_fc_mapping_int(input_2D):
    input_mapped_positive = act_fc_mapping(input_2D)
    input_mapped_negative = act_fc_mapping(input_2D * (-1))
    input_mapped = np.hstack((input_mapped_positive, input_mapped_negative))
    return input_mapped

def act_conv_mapping_int(input_act, weight_4D, stride):
    input_mapped_positive = act_conv_mapping(input_act, weight_4D, stride)
    input_mapped_negative = act_conv_mapping(input_act * (-1), weight_4D, stride)
    input_mapped = np.hstack((input_mapped_positive, input_mapped_negative))
    return input_mapped

