import numpy as np
def batchnorm(input_act, beta, gamma, moving_average, moving_variance, eps=0.001):
	input_shape=input_act.shape
	output_act=np.zeros((input_shape[0],input_shape[1], input_shape[2]))
	for i in range(len(input_act)):
		scale = gamma[i] / np.sqrt(moving_variance[i] + eps)
		output_act[i] = input_act[i] * scale + (beta[i] - moving_average[i] * scale)
	return output_act

def ReLu(input_act):
	input_act[input_act<0] = 0
	return input_act

