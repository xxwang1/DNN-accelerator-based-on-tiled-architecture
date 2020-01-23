#### cb_size=32
import numpy as np
import math
import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set, mapping_mode
from Crossbar import Crossbar
# --- Generate test arrays ---
# Array for fc layers
array_test = np.array(list(range(78400))) #7840
array_test = np.reshape(array_test, (784,100)) #(784, 10)

# Array for conv layer, test filter weight with dimension (3, 3, 4, 5).
# Note that the dimension format is as (row, col, ch, filter)
#list_final ends with 180 entries
list_final = []
for idx_el in range(9):
    list_temp = []
    for idx_f in range(5):
        list_temp.append(1+idx_el + (idx_f+1)*10)
    list_final += list_temp*4

#creates a 3x3x4x5 4-dimensional array out of list_final 
np_final = np.array(list_final)
np_final = np_final.reshape(3,3,4,5)
# ----------------------------

weights = {'fc1': array_test, 'conv1': np_final} 
weights_mapped = {}

def get_weight_to_map(weight_2D, mode=mapping_mode):
    if mode == 'neg_act':

        #creates 2 separate arrays to split up the postive and negative values of weight_2D
        weight_pos = process_weight(weight_2D, True)
        weight_neg = process_weight(weight_2D, False)
        
        #new array with twice the number of rows of weight_2D but same number of columns
        weight2d = np.zeros((weight_2D.shape[0]*2, weight_2D.shape[1])) 
        #alternates the rows of weight2d with 1 weight_pos row and 1 weight_neg row etc 
        weight2d[::2] = weight_pos
        weight2d[1::2] = weight_neg
        #replaces all -0.0 in weight2d with +0
        weight2d[weight2d == -0.0] = +0 
        weight_to_map = weight2d
        return weight_to_map
    elif mode == 'ideal' or mode == None or mode == 'dual_array':

        #creates 2 separate arrays to split up the postive and negative values of weight_2D 
        weight_positive = process_weight(weight_2D, True)
        weight_negative = process_weight(weight_2D, False)

        return weight_positive, weight_negative
        

    


def fc_mapping_base(weight): # Get a 2D weight return a 2D array map
    w_shape = weight.shape
    cb_array_row_cnt = math.ceil(w_shape[0]/cb_size[0]) #w_shape[0] is the number of rows of weight, cb_size[0] is the first dimension of cb_size from gl.py
    cb_array_col_cnt = math.ceil(w_shape[1]/cb_size[1]) #w_shape[1] is the number of columns of weight, cb_size[0] is the second dimension of cb_size from gl.py
    
    mapped_array = np.empty([cb_array_row_cnt, cb_array_col_cnt])   #sets empty array with previous dimensions
    
    #loops through each element in mapped_array
    for cb_row in range(cb_array_row_cnt): 
        # Set the range of the array slice by defining starting and ending row
        row_strt = cb_row*cb_size[0]
        if cb_row == cb_array_row_cnt-1: # execute if last row
            row_end = -1
        else:
            row_end = (cb_row+1)*cb_size[0]
            
        for cb_col in range(cb_array_col_cnt):
            crossbar_set.append(Crossbar())
            
            # Set the range of the array slice by defining starting and ending col
            col_strt = cb_col*cb_size[1]
            if cb_col == cb_array_col_cnt-1:    # execute if last column
                col_end = -1
            else:
                col_end = (cb_col+1)*cb_size[1]
            
            # Create new crossbar object and update the array stored with sliced weight array
            if row_end == -1 and col_end == -1: #if last row and last column
                crossbar_set[-1].update_array(weight[row_strt:, col_strt:])
            elif row_end == -1 and col_end != -1:   #if last row but not last column
                crossbar_set[-1].update_array(weight[row_strt:, col_strt:col_end])
            elif row_end != -1 and col_end == -1:   #if last column but not last row
                crossbar_set[-1].update_array(weight[row_strt:row_end, col_strt:])
            else:   #if neither last row nor last column
                crossbar_set[-1].update_array(weight[row_strt:(row_end), col_strt:(col_end)])
           
            new_cb_id = len(crossbar_set)-1
            mapped_array[cb_row][cb_col] = new_cb_id
    return mapped_array

def fc_mapping(weight_in, mode=mapping_mode):
    
    #gets dimensions of weight_fc_shape and sets the 3rd and 4th dimensions to be the size of weight_2D
    weight_fc_shape = weight_in.shape
    weight_2D = weight_in.reshape(weight_fc_shape[2],weight_fc_shape[3])

    if mode == 'dual_array':
        weight_positive, weight_negative = get_weight_to_map(weight_2D, mode=mode)
        return np.concatenate((fc_mapping_base(weight_positive), fc_mapping_base(weight_negative)))
    elif mode == 'neg_act':
        weight_to_map = get_weight_to_map(weight_2D, mode=mode)
        return fc_mapping_base(weight_to_map)

def conv_mapping(weight, mode=mapping_mode): # Get a 4D weight return a 2D array map)
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    weight_reorder = np.moveaxis(weight, -1, 0)
    w_shape = weight_reorder.shape # (NF, row, col, ch)
    weight_2D = np.reshape(weight_reorder, (w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]))
    weight_2D = np.swapaxes(weight_2D, 0, 1) # formate (row, col); (NF, all el in one position stacked up)
    if mode == 'dual_array':
        weight_positive, weight_negative = get_weight_to_map(weight_2D, mode=mode)
        return np.concatenate((fc_mapping_base(weight_positive), fc_mapping_base(weight_negative)))
    elif mode == 'neg_act':
        weight_to_map = get_weight_to_map(weight_2D, mode=mode)
        return fc_mapping_base(weight_to_map)
    

def fc_mapping_int(weight, mode=mapping_mode):

    #creates 2 separate arrays to split up the postive and negative values of weight 
    weight_positive = process_weight(weight, True)
    weight_negative = process_weight(weight, False)

    mapped_array_positive = fc_mapping(weight_positive)
    mapped_array_negative = fc_mapping(weight_negative)
    mapped_array = np.concatenate((mapped_array_positive, mapped_array_negative))
    return mapped_array

def conv_mapping_int(weight, mode=mapping_mode):

    #creates 2 separate arrays to split up the postive and negative values of weight 
    weight_positive = process_weight(weight, True)
    weight_negative = process_weight(weight, False)

    mapped_array_positive = conv_mapping(weight_positive)
    mapped_array_negative = conv_mapping(weight_negative)
    mapped_array = np.concatenate((mapped_array_positive, mapped_array_negative))
    return mapped_array



#takes a weight array and returns an array of identical dimensions
#the new array contains EITHER the positive or negative values of the original weight in the corresponding positions and has 0s everywhere else  
#needs_positive is either True or False
def process_weight(weight, needs_positive):
    
    #create the new array
    new_weight = np.zeros_like(weight)

    
    if (needs_positive): #execute if positive values of weight are needed
        new_weight[weight>0] = weight[weight>0]
    else:    #execute if negative values of weight are needed
        new_weight[weight<0] = weight[weight<0]
        #turns all negative values to positive 
        new_weight = new_weight * (-1)
    return new_weight


# a = np.full([3,3], 3)
# b = np.full([3,3], -3)
# array = np.stack([a,b])
# print(fc_mapping_symmetric(array))