import numpy as np
import matplotlib.pyplot as plt
from math import log, exp

######################################################################################
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
######################################################################################
class Crossbar_diff():
    #array[row, col]

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

    def get_dots(self, input):
        #input vector as 1D numpy array pulse widths
        output = np.array(np.dot(self.I_array, input)) #charge as output
        return output


######################################################################################
#Parameter: Algebra model
G_min_avg = 2e-6  #average minimal conductance
G_max_avg = 4e-6  #average maximum conductance
P_max = 25
A_avg = P_max/2
B_avg = (G_max_avg - G_min_avg)/(1-exp(-P_max/A_avg))
V_read = 1

######################################################################################
class Crossbar_alg():
    @staticmethod
    def device_update(G_in, G_min, G_max, A, num_pulse): 
        #num_pulse > 0 LTP, num_pulse < 0 LTD
        #Return updated G value
        G = 0
        P = 0
        B = (G_max - G_min)/(1-exp(-P_max/A))
        if num_pulse > 0: #LTP
            B = (G_max - G_min)/(1-exp(-P_max/A))
            P = -A*log(1-(G_in-G_min)/B)
            if P + num_pulse >= P_max:
                P = P_max
            else:
                P = P + num_pulse
            G = B*(1-exp(-P/A)) + G_min
        elif num_pulse == 0:
            return G_in
        else: #LTD
            P = -A*log(1-(G_max-G_in)/B)
            if P + abs(num_pulse) >= P_max:
                P = P_max
            else:
                P = P + abs(num_pulse)
            G = -B*(1-exp(-P/A)) + G_max

        return G

    def __init__(self, cb_size, new_array):
        self.cb_size = cb_size # CB size
        self.G_min_array = np.full((cb_size, cb_size), G_min_avg)
        self.G_max_array = np.full((cb_size, cb_size), G_max_avg)
        self.A_array = np.full((cb_size, cb_size), A_avg)
        self.G_array = np.full((cb_size, cb_size), G_min_avg)
        for ix, iy in np.ndindex(self.G_array.shape):
            self.G_array[ix,iy] = self.device_update(self.G_array[ix,iy], self.G_min_array[ix,iy], self.G_max_array[ix,iy], self.A_array[ix,iy], new_array[ix,iy])

    def update(self, new_array):
        for ix, iy in np.ndindex(self.G_array.shape):
            self.G_array[ix,iy] = self.device_update(self.G_array[ix,iy], self.G_min_array[ix,iy], self.G_max_array[ix,iy], self.A_array[ix,iy], new_array[ix,iy])

    def get_array(self):
        return self.G_array

    def accumulate(self, input): 
        output = np.matmul(input*V_read, self.G_array)
        return output


######################################################################################
#Device Quantization Model
G_stdev = 10    #Assume conductance normalized to 8-bit
num_bits = 8    #number of bits for ADC and input activation
######################################################################################
class Crossbar_Quant():
    def __init__(self, cb_size, new_array):
        self.cb_size = cb_size
        self.G_array = new_array + np.random.normal(0, G_stdev, (cb_size, cb_size))

    def accumulate(self, input):
        output = np.matmul(input, self.G_array)
        output = output/(self.cb_size*2**num_bits*2**num_bits)*2**num_bits
        output = np.round(output)
        return output

    def get_array(self):
        return self.G_array




######################################################################################
# Test code
######################################################################################
def CB_alg_test():
    size = 5

    #ini_array = np.array(range(size*size)).reshape(size, size)
    #test_array = np.array(ini_array)
    #cb_test = Crossbar_diff(size, test_array)
    #output = cb_test.get_array().flatten()

    ini_array = np.array(range(size*size)).reshape(size, size)
    cb_test = Crossbar_alg(size, ini_array)

    output = cb_test.get_array().flatten()
    print(output)
    #plt.plot(output)
    #plt.show()

    cb_test2 = Crossbar_alg(size, np.full((size, size),P_max))
    cb_test2.update(-1*ini_array)
    output2 = cb_test2.get_array().flatten()
    print(output)
    #plt.plot(output2)
    #plt.show()

    plt.plot(np.concatenate((output, output2))*1e6, color='red',)
    plt.xlabel("#Pulse")
    plt.ylabel(r'Condcutance ($\mu$S)')
    plt.show()

def CB_quant_test():
    size = 32
    ini_array = np.full((size, size), 100)
    cb_test = Crossbar_Quant(size, ini_array)
    activation = np.full(size, 100)
    output = cb_test.accumulate(activation)
    print(output)
    plt.plot(output)
    plt.show()

CB_quant_test()