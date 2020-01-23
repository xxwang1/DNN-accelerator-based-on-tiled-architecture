from Weight_mapping import *
from Activation_mapping import *
import numpy as np
<<<<<<< HEAD
import os

#os.system('taskset -p 0xffffffff %d' % os.getpid())
=======
>>>>>>> 85e788daca47d9475be948ad187b9c2cbd8713ba
def fc_mapping_symmetric(weight):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    weight_positive = np.zeros_like(weight)
    weight_negative = np.zeros_like(weight)
    weight_positive[weight>0] = weight[weight>0]
    weight_negative[weight<0] = weight[weight<0]
    mapped_array_positive = fc_mapping(weight_positive)
    mapped_array_negative = fc_mapping(weight_negative)
    mapped_array = np.stack((mapped_array_positive, mapped_array_negative))
    return mapped_array

def conv_mapping_symmetric(weight):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    weight_positive = np.zeros_like(weight)
    weight_negative = np.zeros_like(weight)
    weight_positive[weight>0] = weight[weight>0]
    weight_negative[weight<0] = weight[weight<0]
    mapped_array_positive = conv_mapping(weight_positive)
    mapped_array_negative = conv_mapping(weight_negative)
    mapped_array = np.stack((mapped_array_positive, mapped_array_negative))
    return mapped_array



def act_fc_mapping_symmetric(input_2D):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    input_mapped_positive = act_fc_mapping(input_2D)
    input_mapped_negative = act_fc_mapping(input_2D)
    input_mapped = np.hstack((input_mapped_positive, input_mapped_negative))
    return input_mapped

def act_conv_mapping_symmetric(input_act, weight_4D, stride):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    input_mapped_positive = act_conv_mapping(input_act, weight_4D, stride)
    input_mapped_negative = act_conv_mapping(input_act, weight_4D, stride)
    input_mapped = np.hstack((input_mapped_positive, input_mapped_negative))
    return input_mapped


