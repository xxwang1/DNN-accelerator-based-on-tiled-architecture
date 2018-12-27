import numpy as np
import matplotlib.pyplot as plt

#gmin = 1.5
#gmax = 3
#Pmax = 100
#A_LTP = 0.5

#P_LTP = 0
#P_LTD = 1
#g = (gmax - gmin)/(1-np.exp(-Pmax/A_LTP))

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

def device_update(w, num_pulse):
    if num_pulse > 0:
        for i in range(abs(num_pulse)):
            w = w + B_p*(1-w)*pulse_wdith
    else:
        for i in range(abs(num_pulse)):
            w = w + B_n*w*pulse_wdith

    return w

w = 0.003

w_out = device_update(w, 10)
print(w_out)