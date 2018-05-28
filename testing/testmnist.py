# encoding: utf-8

"""

derp

"""

import mnist
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

dir_mnist = r'D:\projectdata\mnsit'


image_list, label_list = mnist.train_images(), mnist.train_labels()

i_index = 0

image = image_list[i_index]
label = label_list[i_index]
pixels = np.array(image, dtype='uint8')


sgm = a = 2
blur_image = scipy.ndimage.filters.gaussian_filter(pixels, a)

result_orig = scipy.ndimage.sobel(pixels)
result_blur = scipy.ndimage.sobel(blur_image)

plt.imshow(pixels)
plt.show()
plt.imshow(blur_image)
plt.show()
plt.imshow(result_orig)
plt.show()
plt.imshow(result_blur)
plt.show()

points = np.nonzero(result_orig)
points = np.array(list(zip(points[0], points[1])))
k
from scipy.spatial import ConvexHull
hull = ConvexHull(points)
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()