import numpy as np
import math
import Layer_function
from Layer_function import fc
from Layer_function import convlution
from Layer_function import dw_convlution
import nonlinear
from nonlinear import batchnorm
from nonlinear import ReLu
import img_TFRecords
from img_TFRecords import image_to_act
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping
from Weight_mapping import dw_conv_mapping
import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set

import tensorflow as tf
import numpy as np
import os
import h5py
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.keras import layers





#设置使用指定GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#下面这段代码是在训练好之后将所有的权重名字和权重值罗列出来，训练的时候需要注释掉
reader = tf.train.NewCheckpointReader('./mobilenet_v1_1.0_224.ckpt')
variables = reader.get_variable_to_shape_map()
#define weights
for ele in variables:
	if ele.endswith('weights'):
		if ele.endswith('depthwise_weights'):
			index=ele[12:-18]
			locals()['weight_'+index]=reader.get_tensor(ele)
			shape=reader.get_tensor(ele).shape
			locals()['weight_'+index]=reader.get_tensor(ele).reshape(shape[0],shape[1],shape[2])

		else:
			index = (ele[12:-8])
			locals()['weight_'+index] = reader.get_tensor(ele)
	elif 'BatchNorm' in ele:
		if ele.endswith('beta'):
			index=ele[12:-15]
			locals()['beta_'+index]=reader.get_tensor(ele)
			
		elif ele.endswith('gamma'):
			index=ele[12:-16]
			locals()['gamma_'+index]=reader.get_tensor(ele)
		elif ele.endswith('moving_mean'):
			index=ele[12:-22]
			locals()['moving_mean_'+index]=reader.get_tensor(ele)
		elif ele.endswith('moving_variance'):
			index=ele[12:-26]
			locals()['moving_variance_'+index]=reader.get_tensor(ele)



#layers
#Conv2d_0 input(ch, row, col)
stride_depthwise=np.array([1,2,1,2,1,2,1,1,1,1,1,2,1])
stride_pointwise=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

weight_Conv2d_0_mapped=conv_mapping(weight_Conv2d_0)

input_act=image_to_act()
output_Conv2d_0 = convlution(input_act,weight_Conv2d_0,weight_Conv2d_0_mapped,2)
input_Conv2d_1_depthwise = ReLu(batchnorm(output_Conv2d_0,beta_Conv2d_0,gamma_Conv2d_0,moving_mean_Conv2d_0,moving_variance_Conv2d_0))
print(input_Conv2d_1_depthwise.shape)
for i in range(1,14):
	locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped']=dw_conv_mapping(locals()['weight_Conv2d_'+str(i)+'_depthwise'], stride_depthwise[i-1], 2, 2)
	locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped']=conv_mapping(locals()['weight_Conv2d_'+str(i)+'_pointwise'])
weight_fc_shape=locals()['weight_Logits/Conv2d_1c_1x1'].shape
weight_fc=locals()['weight_Logits/Conv2d_1c_1x1'].reshape(weight_fc_shape[2],weight_fc_shape[3])
weight_fc_mapped=fc_mapping(weight_fc)
for i in range(1,14):
	#weight_depthwise.shape=locals()['weight_Conv2d_'+str[i]+'_depthwise'].shape
	#time_multiplex=
	#print(locals()['input_Conv2d_'+str(i)+'_depthwise'])
	#print(locals()['input_Conv2d_'+str(i)+'_depthwise'].shape)
    locals()['output_Conv2d_'+str(i)+'_depthwise'] = dw_convlution(locals()['input_Conv2d_'+str(i)+'_depthwise'],locals()['weight_Conv2d_'+str(i)+'_depthwise'],locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'],stride_depthwise[i-1],2,2)
    locals()['input_Conv2d_'+str(i)+'_pointwise'] = ReLu(batchnorm(locals()['output_Conv2d_'+str(i)+'_depthwise'],locals()['beta_Conv2d_'+str(i)+'_depthwise'],locals()['gamma_Conv2d_'+str(i)+'_depthwise'],locals()['moving_mean_Conv2d_'+str(i)+'_depthwise'],locals()['moving_variance_Conv2d_'+str(i)+'_depthwise']))
    print(locals()['input_Conv2d_'+str(i)+'_pointwise'].shape)
    
    locals()['output_Conv2d_'+str(i)+'_pointwise'] = convlution(locals()['input_Conv2d_'+str(i)+'_pointwise'], locals()['weight_Conv2d_'+str(i)+'_pointwise'], locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'], stride_pointwise[i-1])
    locals()['input_Conv2d_'+str(i+1)+'_depthwise'] = ReLu(batchnorm(locals()['output_Conv2d_'+str(i)+'_pointwise'],locals()['beta_Conv2d_'+str(i)+'_pointwise'],locals()['gamma_Conv2d_'+str(i)+'_pointwise'],locals()['moving_mean_Conv2d_'+str(i)+'_pointwise'],locals()['moving_variance_Conv2d_'+str(i)+'_pointwise']))
    print(locals()['input_Conv2d_'+str(i+1)+'_depthwise'].shape)

#Average pooling
input_fc=np.sum(input_Conv2d_14_depthwise,axis=1)
input_fc=np.sum(input_fc,axis=1)
input_fc_shape=input_fc.shape
input_fc=input_fc.reshape(1,input_fc_shape[0])
print(input_fc.shape)
#fc layer


output_fc=fc(input_fc,weight_fc,weight_fc_mapped)
print(output_fc.shape)
#softmax
result=tf.nn.softmax(output_fc)
with tf.Session() as Sess:
	result=result.eval(session=Sess)
print(result)