from PIL import Image
from stl_tools import numpy2stl

from scipy.misc import imresize
from scipy.ndimage import gaussian_filter

import numpy as np


A = np.array(Image.open('/home/charmmaria/Pictures/IMG_20180513_131704.jpg').convert('LA'))
A = np.array(Image.open('/home/charmmaria/Pictures/IMG_20180702_091844.jpg').convert('LA'))
temp_shape = A.shape
A = imresize(A[:, :, 0], (256, 256))  # load Lena image, shrink in half
A = gaussian_filter(A, 1)  # smoothing
A.shape
A = np.reshape(A, temp_shape[0:2])
numpy2stl(A, "/home/charmmaria/Pictures/temp_black_mich.stl", scale=0.1, solid=False)