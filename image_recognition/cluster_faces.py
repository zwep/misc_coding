#clustering_faces


# =========================================================================== #
#	Libraries
# =========================================================================== #

import os
import sys

import pandas as pd
import numpy as np

# Loading image with PIL
from PIL import Image
# Loading image with scipy
import scipy.ndimage as scim

# Plotting libraries
import matplotlib.pyplot as plt

# =========================================================================== #
#	Locations
# =========================================================================== #

loc_faces = r"C:\Users\C35612.LAUNCHER\Testing_data\Webcam\Image"

# =========================================================================== #
#	Load Data
# =========================================================================== #

os.chdir(loc_faces)

# Load JPG
A = scim.imread("webcam_face_39.0.jpg")
plt.imshow(A)
plt.show()

# Some transformation...
B = scim.prewitt(A)
plt.imshow(B)
plt.show()

# Some transformation...
B = scim.sobel(A,2)
plt.imshow(B)
plt.show()
sx = scim.sobel(A, axis=0, mode='constant')
sy = scim.sobel(A, axis=1, mode='constant')

sob = np.hypot(sx, sy)
plt.imshow(sob[0:1,0:1,:])
plt.show()


im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1

im = scim.rotate(im, 15, mode='constant')
im = scim.gaussian_filter(im, 8)
sx = scim.sobel(im, axis=0, mode='constant')
sy = scim.sobel(im, axis=1, mode='constant')
sob = np.hypot(sx, sy)

sx_A = sx[0:3,0:3,:]
sy_A = sy[0:3,0:3,:]
sob_A = np.hypot(sx_A, sy_A)
plt.imshow(sob_A)
plt.show()

sob = np.hypot(sx, sy)
plt.imshow(sob)
plt.show()

# Some transformation...
B = scim.uniform_filter(A)
plt.imshow(B)
plt.show()

# Some transformation...
B = scim.rank_filter(A,5,10)
plt.imshow(B)
plt.show()

# Some transformation...
B = scim.fourier.fourier_uniform(A,[10,15,20])
plt.imshow(B)
plt.show()


# Convolution, let's see if I understand it...
# Yes I do understand it now
# But not completely
ONE = np.ones((98,98))
ZERO = np.zeros((98,98))
C = np.stack([ZERO,EYE,ZERO],axis = 2)

D = scim.convolve(A[:,:,1],C[:,:,1])
plt.imshow(D)
plt.show()

plt.imshow(A[:,:,1])
plt.show()


# Extract features... eigenfaces?
A = Image.open("webcam_face_39.0.jpg")
img = Image.fromarray(A, 'RGB')
# More options?


# Get the integral image for Haar transformation
integral_im_A = np.cumsum(np.cumsum(A,axis = 0),axis = 1)

# How to determine window size?
# And how to determine the features... and how to classify/



