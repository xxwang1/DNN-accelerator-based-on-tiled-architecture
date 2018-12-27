import numpy as np
import matplotlib.pyplot as plt

n1 = 9e-8
n2 = 15.5
gamma = 1e-5
delta = 4
pulse_wdith = 100e-6

Vw = 1.5
Vr = 0.5

A = (gamma*np.sinh(delta*Vr))
B = n1*np.sinh(n2*Vw)

num_point = 20

w = 0.03
gmax = 10
I_plot = np.zeros(num_point)

for i in range(num_point):
    I_plot[i] = w*A
    w = w + B*(1-w)*pulse_wdith

    #I_plot[i] = w*(gamma*np.sinh(delta*Vr))
    #w = w + n1*np.sinh(n2*Vw)*(1-w)*pulse_wdith

plt.plot(I_plot*1e6)
plt.show