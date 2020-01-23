import numpy as np
import tensorflow as tf
import os
# os.system("taskset -p 0xff %d" % os.getpid())

def batchnorm(input_act, beta, gamma, moving_average, moving_variance, eps=0.001):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	# input_act = np.moveaxis(input_act, 0, -1)
	# input_tensor = tf.convert_to_tensor(input_act)
	# output_tensor = tf.nn.batch_normalization(input_tensor, moving_average, moving_variance, beta, gamma, variance_epsilon=0.001)
	# sess = tf.Session()
	# output_act = output_tensor.eval(session = sess)
	# output_act = np.moveaxis(output_act, -1, 0)
	Ch, N, N = input_act.shape
	sigma = np.sqrt(moving_variance + eps)
	sigma = np.tile(sigma.reshape(Ch, 1, 1), (1, N, N))
	gamma = np.tile(gamma.reshape(Ch, 1, 1), (1, N, N))
	beta = np.tile(beta.reshape(Ch, 1, 1), (1, N, N))
	moving_average = np.tile(moving_average.reshape(Ch, 1, 1), (1, N, N))
	out = gamma * (input_act - moving_average) / sigma + beta
	out = out.reshape(input_act.shape)
	#output_act=np.zeros((input_shape[0],input_shape[1], input_shape[2]))
	#for i in range(len(input_act)):
	#	scale = gamma[i] / np.sqrt(moving_variance[i] + eps)
	#	output_act[i] = input_act[i] * scale + (beta[i] - moving_average[i] * scale)
	return out

def ReLu(input_act):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	input_act[input_act<0] = 0	#all values of input_act less than zero get set to 0
	# input_act[input_act>6] = 6	#all values of input_act greater than 6 get set to 6
	return input_act

def Relu6_quant(input_act, scale):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	input_act[input_act<0] = 0	#all values of input_act less than zero get set to 0
	input_act[input_act>6] = 6	#all values of input_act greater than 6 get set to 6
	# return input_act
	return np.round(input_act/scale)

def Relu8_quant(input_act, scale):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	input_act[input_act<0] = 0	#all values of input_act less than zero get set to 0
	input_act[input_act>8] = 8	#all values of input_act greater than 8 get set to 8
	# return input_act
	return np.round(input_act/scale)

def Relu6(input_act):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	input_act[input_act<0] = 0	#all values of input_act less than zero get set to 0
	input_act[input_act>6] = 6 #all values of input_act greater than 6 get set to 6
	# return input_act
	return input_act