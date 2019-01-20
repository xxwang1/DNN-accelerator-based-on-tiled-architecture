import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

t = np.arange(0, 10, 1/100)
v = 0.5*signal.square((t-1)*(6/10), 0.2)

plt.plot(t,v)
plt.axis('off')
plt.figure(figsize=(2,4))
plt.savefig('fig1.svg')
plt.show()