import tensorflow as tf
import numpy as np
import os
import h5py
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.keras import layers
import Layer_function
from Layer_function import dw_convlution
from Layer_function import convlution
from Layer_function import pw_convlution
from Layer_function import fc
import Weight_mapping
from Weight_mapping import dw_conv_mapping
from Weight_mapping import conv_mapping
from Weight_mapping import pw_conv_mapping
from Weight_mapping import fc_mapping

input_act=np.array(list(range(50)))
input_act=input_act.reshape(1,50)
weight=np.array(list(range(2500)))
weight=weight.reshape(50,50)
weight_mapped=fc_mapping(weight)
#output=pw_convlution(input_act, weight, weight_mapped, 1)
#print(output)
#print(output.shape)

#weight_mapped_conv=conv_mapping(weight)
#output_conv=convlution(input_act, weight, weight_mapped_conv, 1)
#print(output_conv-output)
#print(output_conv.shape)
print(input_act)
print(weight)
output=fc(input_act, weight, weight_mapped)
print(output)
print(output.shape)
o1=0
for i in range(50):
	o1+=i*(i*50+3)
print(o1)
#for i in range(5):
#	input_x=input_act[i].reshape(1,10,10)

#	weight_x=weight[:,:,i:i+1]
#	weight_x=weight_x.reshape(3,3,1,1)
#	weight_x_mapped=conv_mapping(weight_x)
#	output_x=convlution(input_x, weight_x, weight_x_mapped, 1)
#	print(i)
#	print(input_x)
#	print(weight_x)
#	print(output_x)

