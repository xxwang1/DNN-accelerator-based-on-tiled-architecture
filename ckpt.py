import tensorflow as tf
import numpy as np
import Weight_mapping
from Weight_mapping import fc_mapping
from Weight_mapping import conv_mapping

from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set

reader = tf.train.NewCheckpointReader('../vgg_16.ckpt')
variables = reader.get_variable_to_shape_map()
# #define weights

# for ele in variables:
# 	f=open('variable.txt','a+')
# 	f.write(ele + '\n')
# 	f.close()
fp = open('variable.txt',"r+")
linesText = fp.readlines()
linesText.sort()
# for line in linesText:
#     print(line)
fp.seek(0)
fp.writelines(linesText)
fp.close()
# f = open('variable_quant.txt')
# lines = f.readlines()
# f.close()
# lines.sort()
# for line in lines:
# 	f=open('quant.txt','a+')
# 	f.write(line)
# 	f.close()
# f=open('variable.txt')
# lines = f.readlines()
# f.close()
weight = {}
bias = {}
layer = 0
parameters = 0
for ele in linesText:
    if ele.endswith('weights\n'):
        if layer <= 12:
            weight['conv_'+ str(layer)] = reader.get_tensor(ele[:-1])
            layer = layer + 1
            # print(ele)
            # w_shape = weight['conv_'+ str(layer-1)].shape
            # print(weight['conv_'+ str(layer-1)].shape)
            # parameters = w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3]
            # print('parameters=', parameters)
            # print('\n')
        else:
            weight['fc_' + str(layer)] = reader.get_tensor(ele[:-1])
            layer = layer + 1
            # print(ele)
            # print(weight['fc_'+ str(layer-1)].shape)
            # w_shape = weight['fc_'+ str(layer-1)].shape
            # parameters = w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3]
            # print('parameters=', parameters)
            # print('\n')
layer = 0
for ele in linesText:
    if ele.endswith('biases\n'):
        if layer <= 12:
            bias['conv_'+ str(layer)] = reader.get_tensor(ele[:-1])
            layer = layer + 1
            # print(ele)
            # print(bias['conv_'+ str(layer-1)].shape)
        else:
            bias['fc_' + str(layer)] = reader.get_tensor(ele[:-1])
            layer = layer + 1
            # print(ele)
            # print(bias['fc_'+ str(layer-1)].shape)
    # if ele.endswith('rgb\n'):
        # print(reader.get_tensor(ele[:-1]))
# print('num of parameters=', parameters)
        
	# 	print(ele[:-1])
	# 	print(reader.get_tensor(ele[:-1]))
	# 	print(reader.get_tensor(ele[:-1]).shape)
	# 	if ele.endswith('Conv2d_1_pointwise/weights\n'):
	# 		print(np.max(reader.get_tensor(ele[:-1])))
	# 		print(np.min(reader.get_tensor(ele[:-1])))
	# 		quant = np.round((reader.get_tensor(ele[:-1]) + 3.6615257) / (4.0653954 + 3.6615257) * 255)
	# 		print(np.max(quant))
	# 		print(np.min(quant))
	# if ele.endswith('beta\n') or ele.endswith('gamma\n') or ele.endswith('moving_mean\n') or ele.endswith('moving_variance\n') or ele.endswith('biases\n'):
	# 	print(ele[:-1])
	# 	print(reader.get_tensor(ele[:-1]))
	#if ele.endswith('biases'):
		#print(ele)
		#print(reader.get_tensor(ele))
		#print(reader.get_tensor(ele).shape)
# 	if ele.endswith('weights'):
# 		if ele.endswith('depthwise_weights'):
# 			index=ele[12:-18]
# 			locals()['weight_'+index]=reader.get_tensor(ele)
# 			shape=reader.get_tensor(ele).shape
# 			locals()['weight_'+index]=reader.get_tensor(ele).reshape(shape[0],shape[1],shape[2])
# 		else:
# 			#print(ele)
# 			index = (ele[12:-8])
# 			locals()['weight_'+index] = reader.get_tensor(ele)
# 	elif 'BatchNorm' in ele:
# 		if ele.endswith('beta'):
# 			index=ele[12:-15]
# 			locals()['beta_'+index]=reader.get_tensor(ele)
			
# 		elif ele.endswith('gamma'):
# 			index=ele[12:-16]
# 			locals()['gamma_'+index]=reader.get_tensor(ele)
# 		elif ele.endswith('moving_mean'):
# 			index=ele[12:-22]
# 			locals()['moving_mean_'+index]=reader.get_tensor(ele)
# 		elif ele.endswith('moving_variance'):
# 			index=ele[12:-26]
# 			locals()['moving_variance_'+index]=reader.get_tensor(ele)
# weight_Conv2d_0_mapped=conv_mapping(locals()['weight_Conv2d_0'])
# #print('weight_Conv2d_0')
# #print(locals()['weight_Conv2d_0'])
# stride_depthwise=np.array([1,2,1,2,1,2,1,1,1,1,1,2,1])
# stride_pointwise=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

# for i in range(1,14):
# 	locals()['weight_Conv2d_'+str(i)+'_depthwise_mapped']=dw_conv_mapping(locals()['weight_Conv2d_'+str(i)+'_depthwise'], stride_depthwise[i-1], 2, 2)
# 	locals()['weight_Conv2d_'+str(i)+'_pointwise_mapped']=pw_conv_mapping(locals()['weight_Conv2d_'+str(i)+'_pointwise'])
# 	#print('weight_Conv2d_'+str(i)+"_depthwise")
# 	#print(locals()['weight_Conv2d_'+str(i)+"_depthwise"].shape)
# 	#print('weight_Conv2d_'+str(i)+'_pointwise')
# 	#print(locals()['weight_Conv2d_'+str(i)+'_pointwise'].shape)
# weight_fc_shape=locals()['weight_Logits/Conv2d_1c_1x1'].shape
# #print('weight_Logits/Conv2d_1c_1x1')
# #print(locals()['weight_Logits/Conv2d_1c_1x1'].shape)
# weight_fc=locals()['weight_Logits/Conv2d_1c_1x1'].reshape(weight_fc_shape[2],weight_fc_shape[3])
# #print(weight_fc.shape)
# weight_fc_mapped=fc_mapping(weight_fc)
# a=np.random.random((3,3,5))
# b=np.moveaxis(a,-1,0)
# #print(a)
# #print(b)