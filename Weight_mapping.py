#### cb_size=32
import numpy as np
import math
import Crossbar
from Crossbar import Crossbar
import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set
# --- Generate test arrays ---
# Array for fc layers
array_test = np.array(list(range(78400))) #7840
array_test = np.reshape(array_test, (784,100)) #(784, 10)

# Array for conv layer, test filter weight with dimention (3, 3, 4, 5).
# Note that the dimention format is as (row, col, ch, filter)
list_final=[]
for idx_el in range(9):
    list_temp = []
    for idx_f in range(5):
        list_temp.append(1+idx_el+(idx_f+1)*10)
    list_final+=list_temp*4

np_final = np.array(list_final)
np_final = np_final.reshape(3,3,4,5)
# ----------------------------
 
weights = {'fc1': array_test, 'conv1': np_final} 
weights_mapped = {}

def fc_mapping(weight): # Get a 2D weight return a 2D array map
    w_shape = weight.shape
    cb_array_row_cnt = math.ceil(w_shape[0]/cb_size)
    cb_array_col_cnt = math.ceil(w_shape[1]/cb_size)
    
    mapped_array = np.empty([cb_array_row_cnt, cb_array_col_cnt])
    
    for cb_row in range(cb_array_row_cnt): 
        # Set the range of the array slice by defining staring and ending row
        row_strt = cb_row*cb_size
        if cb_row == cb_array_row_cnt-1: # check if last row
            row_end = -1
        else:
            row_end = (cb_row+1)*cb_size 
            
        for cb_col in range(cb_array_col_cnt):
            crossbar_set.append(Crossbar())
            
            # Set the range of the array slice by defining staring and ending col
            col_strt = cb_col*cb_size
            if cb_col == cb_array_col_cnt-1:
                col_end = -1
            else:
                col_end = (cb_col+1)*cb_size
            
            # Create new crossbar object and update the array stored with sliced weight array
            if row_end==-1 and col_end==-1:
                crossbar_set[-1].update_array(weight[row_strt:, col_strt:])
            elif row_end==-1 and col_end!=-1:
                crossbar_set[-1].update_array(weight[row_strt:, col_strt:col_end])
            elif row_end!=-1 and col_end==-1:
                crossbar_set[-1].update_array(weight[row_strt:row_end, col_strt:])
            else:
                crossbar_set[-1].update_array(weight[row_strt:(row_end), col_strt:(col_end)])
           
            new_cb_id = len(crossbar_set)-1
            mapped_array[cb_row][cb_col] = new_cb_id
    return mapped_array

def conv_mapping(weight): # Get a 4D weight return a 2D array map
    weight_reorder = np.moveaxis(weight, -1, 0) 
    w_shape = weight_reorder.shape # (NF, row, col, ch)
    weight_2D = np.reshape(weight_reorder, (w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]))
    weight_2D = np.swapaxes(weight_2D, 0, 1) # formate (row, col); (NF, all el in one position stacked up)
    return fc_mapping(weight_2D)       
    
def dw_conv_mapping(weight_3D, stride, duplicate, time_multiplex): #Get a 3D weight return a 2D array map
    #weight(row, col, num)
    w_shape=weight_3D.shape
    NF_tot=w_shape[2]
    row_flt=w_shape[0]
    col_flt=w_shape[1]
    len_flt=row_flt*col_flt
    
    weight_2D=weight_3D.reshape(w_shape[0]*w_shape[1], w_shape[2])
    new_row_tot= (len_flt+stride*row_flt*(duplicate-1))*math.ceil(NF_tot/time_multiplex)
    new_col_tot= NF_tot*duplicate
    
    weight_dw=np.zeros((new_row_tot, new_col_tot))
    for i in range(NF_tot):
        k= i%(math.ceil(NF_tot/time_multiplex))
        for j in range(duplicate):
            for t in range(len_flt):
                weight_dw[(len_flt+stride*row_flt*(duplicate-1))*k+stride*row_flt*j+t][i*duplicate+j]=weight_2D[t][i]
    return fc_mapping(weight_dw)


weight=np.array(list(range(45)))
weight=weight.reshape(3,3,5)
    
    
for layer in weights: 
           
    if str(layer)[:2] == 'fc':
        weights_mapped[layer] = fc_mapping(weights[layer])
    elif str(layer)[:4] == 'conv':
        weights_mapped[layer] = conv_mapping(weights[layer])

dw=dw_conv_mapping(weight, 1, 3, 2)
