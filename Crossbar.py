# import gl
from gl import cb_size, data, energy
from gl import (w_tot, w_int)
from gl import crossbar_set, mapping_mode, onoff_ratio
from gl import input_set, array2bit, scale2bit
import numpy as np

class Crossbar_ideal():
    def __init__(self, cb_size = cb_size, tot_bit = w_tot, int_bit = w_int):
        self.cb_size = cb_size # CB size
        self.array = np.zeros(cb_size) # create 2D array for crossbar set
        self.tot_bit = tot_bit
        self.int_bit = int_bit
    def update_array(self, new_array): # take 2D decimal array
        if new_array.shape == cb_size:
            self.array = new_array
        else:
            new_row_cnt = new_array.shape[0]
            new_col_cnt = new_array.shape[1]
            self.array[:new_row_cnt, : new_col_cnt] = new_array
    def get_array(self):
        return self.array
    def accumulate(self, input_vector, ADC):
        return np.matmul(input_vector, self.array)

######################################################################################
#Device Quantization Model
G_std = 0.0  #Assume conductance normalized to 8-bit %
G_max = 255
G_min_std = 0.0   #as %
if onoff_ratio == 'infinite':
    G_min = 0
elif onoff_ratio == '100':
    G_min = G_max/100
elif onoff_ratio == '1000':
    G_min = G_max/1000
elif onoff_ratio == '10':
    G_min = G_max/10
G_max_std = 0.0   #as %
# num_bits = 8    #number of bits for ADC and input activation
######################################################################################
class Crossbar_Quant():
    def __init__(self, cb_size = cb_size, new_array = None, mode = mapping_mode):
        self.cb_size = cb_size
        self.G_min_array = np.random.normal(G_min, G_min_std, cb_size)
        self.G_max_array = np.random.normal(G_max, G_max_std, cb_size)
        self.mode = mode
        if new_array == None:
            self.G_array = np.zeros(cb_size)
        else:
            scaled_weights = new_array*((self.G_max_array - self.G_min_array)/self.G_max_array)
            self.G_array = self.G_min_array + np.random.normal(scaled_weights, G_std*scaled_weights)
        if self.mode == 'dual_array':
            self.weight_scale = (np.sum(self.G_array, axis=0))
        elif self.mode == 'neg_act':
            self.weight_scale = np.zeros((2, self.cb_size[1]))
            for i in range(int(self.cb_size[0]/2)):
                temp = self.G_array[(i*2)] - self.G_array[i*2+1]
                self.weight_scale[0][temp>0] += temp[temp>0]
                self.weight_scale[1][temp<0] += temp[temp<0] * (-1)
            # self.weight_scale = (np.sum(self.G_array[::2], axis=0)-np.sum(self.G_array[1::2], axis=0))
            # self.weight_scale = np.max(self.weight_scale_pos, axis=0)
        
        # self.weight_scale = (np.sum(self.G_array[::2], axis=0)-np.sum(self.G_array[1::2], axis=0))

    def update_array(self, new_array): # take 2D decimal array
        if new_array.shape == cb_size:
            self.G_array = new_array
        else:
            new_row_cnt = new_array.shape[0]
            new_col_cnt = new_array.shape[1]
            self.G_array[:new_row_cnt, : new_col_cnt] = new_array

        scaled_weights = np.abs(self.G_array*((self.G_max_array - self.G_min_array)/self.G_max_array))
        self.G_array = self.G_min_array + np.random.normal(scaled_weights, G_std*scaled_weights)
        if self.mode == 'dual_array':
            self.weight_scale = (np.sum(self.G_array, axis =0))
        elif self.mode == 'neg_act':
            self.weight_scale = np.zeros((2, self.cb_size[1]))
            for i in range(int(self.cb_size[0]/2)):
                temp = self.G_array[(i*2)] - self.G_array[i*2+1]
                self.weight_scale[0][temp>0] += temp[temp>0]
                self.weight_scale[1][temp<0] += temp[temp<0] * (-1)
            # self.weight_scale = (np.sum(self.G_array[::2], axis=0)-np.sum(self.G_array[1::2], axis=0))
            # self.weight_scale = self.weight_scale_pos
        self.weight_scale_fc = np.max(self.weight_scale)

    def accumulate(self, input_vector, weight_array, input_scale, num_bits, ADC, bit_serial):
        if bit_serial:
            if ADC == False:
                return np.matmul(input_vector, weight_array)
            else:
                flag = 1
                if input_vector.min() < -0.00001:
                    input_vector = input_vector * (-1)
                    flag = -1
                
                input_vector_bin = np.unpackbits(input_vector.astype(np.uint8)[np.newaxis, :, :], axis = 0)
                input_vector_bin[0][input_vector == 256] = 2
                output = np.matmul(input_vector_bin, weight_array)
            
                scale = self.weight_scale
                scale[scale == 0] = 0.001
                output = np.round(output/array2bit(scale[0:output.shape[-1]])*(2**(num_bits)-1))
                output = output[0] * 128 + output[1] * 64 + output[2] * 32 + output[3] * 16 + output[4] * 8 + output[5] * 4 + output[6] * 2 + output[7]
                output[output>(2**(num_bits+1)-1)] = (2**(num_bits+1)-1)
                output[output<-(2**(num_bits+1)-1)] = -(2**(num_bits+1)-1)
                output = output * array2bit(scale[0:output.shape[-1]]) * flag

                return output
        else:
            output = np.matmul(input_vector, weight_array)
            # data['input_activation'].append((input_vector).tolist())
            # data['weight_array'].append((weight_array).tolist())
            if ADC == False:
                return output
            else:
                # output = np.round(output/scale*(2**(num_bits)-1))
                # output = output / (2**(num_bits)-1) * scale
                if self.mode == 'dual_array':
                    scale = self.weight_scale[:output.shape[-1]]
                    scale[scale == 0] = 0.001
                    output = np.round(output/array2bit(scale) / scale2bit(input_scale) * (2**(num_bits)-1))
                    
                    output[output>(2**(num_bits)-1)] = (2**(num_bits)-1)
                    output[output<-(2**(num_bits)-1)] = -(2**(num_bits)-1)
                    output_quant = output * array2bit(scale) * scale2bit(input_scale)
                elif self.mode == 'neg_act':
                    mask = np.abs(input_vector) / input_vector
                    
                    input_vector_bin = np.unpackbits(np.abs(input_vector).astype(np.uint8)[np.newaxis, :, :], axis = 0) * mask
                    energy[-1] += np.sum(np.matmul(input_vector_bin, weight_array) * 0.3 / 255 * 3e-6 / 5e6)
                    scale = self.weight_scale[:, :output.shape[-1]]
                    scale[scale == 0] = 0.001
                    output_pos = np.zeros_like(output)
                    output_neg = np.zeros_like(output)
                    output_pos[output>0] = output[output>0]
                    output_neg[output<0] = output[output<0]
                    # for i in range(output.shape[-1]):
                    output_pos = np.round(output_pos/array2bit(scale[0]) / scale2bit(input_scale) * (2**(num_bits)-1))
                    output_neg = np.round(output_neg/array2bit(scale[1]) / scale2bit(input_scale) * (2**(num_bits)-1))
                    output_pos[output_pos>(2**(num_bits)-1)] = (2**(num_bits)-1)
                    output_neg[output_neg<-(2**(num_bits)-1)] = (2**(num_bits)-1)
                    # output = np.round(output/array2bit(scale) / scale2bit(input_scale) * (2**(num_bits)-1))
                    # data['output_activation'].append((output).tolist())
                    # output[output>(2**(num_bits+1)-1)] = (2**(num_bits+1)-1)
                    # output[output<-(2**(num_bits+1)-1)] = -(2**(num_bits+1)-1)
                    output_pos = output_pos * array2bit(scale[0]) * scale2bit(input_scale)
                    output_neg = output_neg * array2bit(scale[1]) * scale2bit(input_scale)
                    output_quant = output_pos + output_neg
                # output = output * array2bit(scale) * scale2bit(input_scale)
        
                return output_quant
        
    def accumulate_dw(self, input_vector, weight_array, index0, index1, input_scale, num_bits, ADC, bit_serial):
        if bit_serial:
            if ADC == False:
                return np.matmul(input_vector, weight_array)
            else:
                flag = 1
                if input_vector.min() < -0.00001:
                    input_vector = input_vector * (-1)
                    flag = -1
                
                input_vector_bin = np.unpackbits(input_vector.astype(np.uint8)[np.newaxis, :, :], axis = 0)
                input_vector_bin[0][input_vector == 256] = 2
                output = np.matmul(input_vector_bin, weight_array)
            
                scale = self.weight_scale
                scale[scale == 0] = 0.001
                output = np.round(output/array2bit(scale[0:output.shape[-1]])*(2**(num_bits)-1))
                output = output[0] * 128 + output[1] * 64 + output[2] * 32 + output[3] * 16 + output[4] * 8 + output[5] * 4 + output[6] * 2 + output[7]
                output[output>(2**(num_bits+1)-1)] = (2**(num_bits+1)-1)
                output[output<-(2**(num_bits+1)-1)] = -(2**(num_bits+1)-1)
                output = output * array2bit(scale[0:output.shape[-1]]) * flag

                return output
        else:
            output = np.matmul(input_vector, weight_array)
            # data['input_activation'].append((input_vector).tolist())
            # data['weight_array'].append((weight_array).tolist())
            if ADC == False:
                return output
            else:
                # output = np.round(output/scale*(2**(num_bits)-1))
                # output = output / (2**(num_bits)-1) * scale
                if self.mode == 'dual_array':
                    # scale = self.weight_scale[:output.shape[-1]]
                    scale = self.weight_scale[index0:index1]
                    scale[scale == 0] = 0.001
                    output = np.round(output/array2bit(scale) / scale2bit(input_scale) * (2**(num_bits)-1))
                    
                    output[output>(2**(num_bits)-1)] = (2**(num_bits)-1)
                    output[output<-(2**(num_bits)-1)] = -(2**(num_bits)-1)
                    output_quant = output * array2bit(scale) * scale2bit(input_scale)
                elif self.mode == 'neg_act':
                    scale = self.weight_scale[:, index0:index1]
                    # scale = self.weight_scale[:, :output.shape[-1]]
                    scale[scale == 0] = 0.001
                    output_pos = np.zeros_like(output)
                    output_neg = np.zeros_like(output)
                    output_pos[output>0] = output[output>0]
                    output_neg[output<0] = output[output<0]
                    # for i in range(output.shape[-1]):
                    output_pos = np.round(output_pos/array2bit(scale[0]) / scale2bit(input_scale) * (2**(num_bits)-1))
                    output_neg = np.round(output_neg/array2bit(scale[1]) / scale2bit(input_scale) * (2**(num_bits)-1))
                    output_pos[output_pos>(2**(num_bits)-1)] = (2**(num_bits)-1)
                    output_neg[output_neg<-(2**(num_bits)-1)] = (2**(num_bits)-1)
                    # # output = np.round(output/array2bit(scale) / scale2bit(input_scale) * (2**(num_bits)-1))
                    # # data['output_activation'].append((output).tolist())
                    # output[output>(2**(num_bits+1)-1)] = (2**(num_bits+1)-1)
                    # output[output<-(2**(num_bits+1)-1)] = -(2**(num_bits+1)-1)
                    output_pos = output_pos * array2bit(scale[0]) * scale2bit(input_scale)
                    output_neg = output_neg * array2bit(scale[1]) * scale2bit(input_scale)
                    output_quant = output_pos + output_neg
                # output = output * array2bit(scale) * scale2bit(input_scale)
        
                return output_quant
        
    def accumulate_fc(self, input_vector, weight_array, input_scale, num_bits, ADC):
        output = np.matmul(input_vector, weight_array)
        if ADC==False:
            return output
        else:
            scale = self.weight_scale_fc*input_scale
            if scale == 0:
                scale = 0.001
        #     if scale ==1:
        #         scale = 1.01
        #     output[output == 0] = 0.01
        #     output_mask = output / np.abs(output)
            
        #     output = np.abs(output)
            output = np.round(output/scale*(2**(num_bits)-1))
            output = output / (2**(num_bits)-1) * scale
        #     output = output * output_mask
        # # output = np.round(output)
            return output
    
    def get_array(self):
        return self.G_array


Crossbar = Crossbar_Quant
