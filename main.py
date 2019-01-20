import numpy as np
import math
import Layer_function
from Layer_function import fc
from Layer_function import convlution
from Layer_function import dw_convlution
from Layer_function import pw_convlution
import nonlinear
from nonlinear import batchnorm
from nonlinear import ReLu
import img_TFRecords

from img_TFRecords import image_to_act
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping
from Weight_mapping import dw_conv_mapping
from Weight_mapping import pw_conv_mapping
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

import os


def MobileNet(start_img,end_img):
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
	#weights mapping
	stride_depthwise=np.array([1,2,1,2,1,2,1,1,1,1,1,2,1])
	stride_pointwise=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

	weight_Conv2d_0_mapped=conv_mapping(locals()['weight_Conv2d_0'])


	for i in range(1,14):
		locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped']=dw_conv_mapping(locals()['weight_Conv2d_'+str(i)+'_depthwise'], stride_depthwise[i-1], 2, 2)
		locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped']=conv_mapping(locals()['weight_Conv2d_'+str(i)+'_pointwise'])
	weight_fc_shape=locals()['weight_Logits/Conv2d_1c_1x1'].shape
	weight_fc=locals()['weight_Logits/Conv2d_1c_1x1'].reshape(weight_fc_shape[2],weight_fc_shape[3])
	weight_fc_mapped=fc_mapping(weight_fc)

	#read images
	
	path ="./ILSVRC2012_img_val/"    #指定需要读取文件的目录
	files =os.listdir(path) #采用listdir来读取所有文件
	files.sort() #排序
	#def MobileNet(start_img, end_img):
	for i in range(start_img,end_img):     #循环读取每个文件名
	    file_=files[i]

	    #for file_ in files:
	    #print(path +file_)
	    if not  os.path.isdir(path +file_):  #判断该文件是否是一个文件夹
	        f_name = str(file_)
	        print(f_name)
	        input_act=image_to_act(f_name)
	        #print(input_act)
	        output_Conv2d_0 = convlution(input_act,locals()['weight_Conv2d_0'],weight_Conv2d_0_mapped,2)
	        #print(output_Conv2d_0)
	        input_Conv2d_1_depthwise = ReLu(batchnorm(output_Conv2d_0,locals()['beta_Conv2d_0'],locals()['gamma_Conv2d_0'],locals()['moving_mean_Conv2d_0'],locals()['moving_variance_Conv2d_0']))
	        print(input_Conv2d_1_depthwise.shape)
	        #print(input_Conv2d_1_depthwise)
	        #print(locals()['gamma_Conv2d_0'].shape)

	        for i in range(1,14):
	        	locals()['output_Conv2d_'+str(i)+'_depthwise'] = dw_convlution(locals()['input_Conv2d_'+str(i)+'_depthwise'],locals()['weight_Conv2d_'+str(i)+'_depthwise'],locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped'],stride_depthwise[i-1],2,2)
	        	locals()['input_Conv2d_'+str(i)+'_pointwise'] = ReLu(batchnorm(locals()['output_Conv2d_'+str(i)+'_depthwise'],locals()['beta_Conv2d_'+str(i)+'_depthwise'],locals()['gamma_Conv2d_'+str(i)+'_depthwise'],locals()['moving_mean_Conv2d_'+str(i)+'_depthwise'],locals()['moving_variance_Conv2d_'+str(i)+'_depthwise']))
	        	print(locals()['input_Conv2d_'+str(i)+'_pointwise'].shape)

	        	locals()['output_Conv2d_'+str(i)+'_pointwise'] = pw_convlution(locals()['input_Conv2d_'+str(i)+'_pointwise'], locals()['weight_Conv2d_'+str(i)+'_pointwise'], locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped'], stride_pointwise[i-1])
	        	locals()['input_Conv2d_'+str(i+1)+'_depthwise'] = ReLu(batchnorm(locals()['output_Conv2d_'+str(i)+'_pointwise'],locals()['beta_Conv2d_'+str(i)+'_pointwise'],locals()['gamma_Conv2d_'+str(i)+'_pointwise'],locals()['moving_mean_Conv2d_'+str(i)+'_pointwise'],locals()['moving_variance_Conv2d_'+str(i)+'_pointwise']))
	        	print(locals()['input_Conv2d_'+str(i+1)+'_depthwise'].shape)

	        #Average pooling
	        print(locals()['input_Conv2d_14_depthwise'])
	        input_fc=np.mean(locals()['input_Conv2d_14_depthwise'],axis=1)
	        input_fc=np.mean(input_fc,axis=1)
	        
	        input_fc_shape=input_fc.shape
	        input_fc=input_fc.reshape(1,input_fc_shape[0])
	        print(input_fc.shape)
	        #fc layer
	        print(input_fc)

	        output_fc=fc(input_fc,weight_fc,weight_fc_mapped)
	        print(output_fc.shape)
	        #softmax
	        output_fc=tf.convert_to_tensor(output_fc)
	        slim = tf.contrib.slim
	        result=tf.nn.softmax(output_fc)
	        print(output_fc)
	        with tf.Session() as Sess:
	        	result=result.eval(session=Sess)
	        	print(result)
	        	f=open('endpoints_max.txt','a+')
	        	f.write(f_name + ' ') # 写入之前的文本中
	        	print(np.max(result), np.argmax(result),file=f)
	        f.close() #看一下列表里的内容
