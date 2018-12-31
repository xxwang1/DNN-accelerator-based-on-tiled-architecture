import numpy as np
import matplotlib.pyplot as plt

#Parameter: Lu Group Differential model
n1 = 9e-8
n2 = 15.5
gamma = 1e-5
delta = 4
pulse_wdith = 100e-6

Vw = 1.5
Vr = 0.5

A = (gamma*np.sinh(delta*Vr))
B_p = n1*np.sinh(n2*Vw)
B_n = n1*np.sinh(n2*(-Vw))

w_avg = 0.03
###########################################

#Parameter: Algebra model

###########################################


class Crossbar_diff():
    @staticmethod
    def device_update(w_in, num_pulse):
        w = w_in
        if num_pulse > 0:
            for i in range(abs(num_pulse)):
                w = w + B_p*(1-w)*pulse_wdith
        else:
            for i in range(abs(num_pulse)):
                w = w + B_n*w*pulse_wdith
        return w

    def __init__(self, cb_size, new_array):
        self.cb_size = cb_size # CB size
        self.w_array = np.full((cb_size, cb_size), w_avg)
        for ix, iy in np.ndindex(self.w_array.shape):
            self.w_array[ix,iy] = self.device_update(self.w_array[ix,iy], new_array[ix,iy])
        self.I_array = self.w_array*A # 2D array for device current

    def update_array(self, new_array): # take 2D decimal array
        for ix, iy in np.ndindex(self.w_array.shape):
            self.w_array[ix,iy] = self.device_update(self.w_array[ix,iy], new_array[ix,iy])
        self.I_array = self.w_array*A # 2D array for device current

    def get_array(self):
        return self.I_array


#class Crossbar_alg():
#    @staticmethod
#    def device_update(w_in, num_pulse):
        
#        return w

#    def __init__(self, cb_size, new_array):
#        self.cb_size = cb_size # CB size
#        self.w_array = np.full((cb_size, cb_size), w_avg)
#        for ix, iy in np.ndindex(self.w_array.shape):
#            self.w_array[ix,iy] = self.device_update(self.w_array[ix,iy], new_array[ix,iy])
        

#    def get_array(self):
#        return self.I_array


    #def matmul(self, input_vector):
    #    return np.matmul(input_vector, self.array)


size = 5

ini_array = np.array(range(size*size)).reshape(size, size)
test_array = np.array(ini_array)
cb_test = Crossbar_diff(size, test_array)
output = cb_test.get_array().flatten()

print(output)
plt.plot(output)
plt.show()
