# encoding: utf-8

"""
Here we test the method for Gaussian Filtering provided by scipy, the convolution operator in numpy and my own creation

Both this answer
https://stackoverflow.com/questions/22669252/how-exactly-does-the-reflect-mode-for-scipys-ndimage-filters-work
As my own question
https://stackoverflow.com/questions/49980292/convolution-with-gaussian-vs-gaussian-filter/49981175?noredirect=1#49981175

Helped to gain the knowledge to create this
"""

import numpy as np
import scipy.stats
import scipy.ndimage
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=160)


m_init = 7
# Create any signal here...
input_signal_init = []  
input_signal_init = np.arange(m_init)
input_signal_init = np.random.choice(range(m_init), m_init)
# Convert to float for better results in scipy.
input_signal_init = np.array(input_signal_init).astype(float)

# Simulating method='reflect'
input_signal = np.array([*input_signal_init[::-1], *input_signal_init, *input_signal_init[::-1]])
# input_signal = input_signal_init
# Define new length of input signal
m = len(input_signal)  
# Properties of the Gaussian
sgm = 2  # dev for standard distr
radius = 4 * sgm
x = np.arange(-radius, radius+1)
n = len(x)
weight_conv = np.zeros(m*(n+m)).reshape(n+m, m)  

# Calculate the gaussian
p = np.polynomial.Polynomial([0, 0, -0.5 / (sgm * sgm)])
input_filter = np.exp(p(x), dtype=np.double)
input_filter /= input_filter.sum()

# Calculate the filter weights
for i in range(weight_conv.shape[1]):
    weight_conv[i:(len(input_filter)+i), i] = input_filter

# My own way of calculating the convolution
self_conv = np.sum(weight_conv * input_signal, axis=1)[(2*m_init+1):(3*m_init+1)]
# Convolution provided by np
np_conv = np.convolve(input_signal, input_filter)[(2*m_init+1):(3*m_init+1)]
# Convolution by scipy with method='reflect'
# !! Here we use the original 'input_signal_init'
scipy_conv = scipy.ndimage.filters.gaussian_filter(input_signal_init, sigma=sgm)

plt.plot(scipy_conv, 'r-')
plt.plot(self_conv, 'bo')
plt.plot(np_conv, 'k.-')
plt.show()

