# encoding: utf-8

"""
Here we are trying the simple-unet architecture proposed by

https://github.com/mirzaevinom/prostate_segmentation

where we have downloaded his weights...
C:\Users\C35612\data\mid\simple_unet_weights.h5

"""

import os
import numpy as np
import pydicom

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage.transform
from experiment.mid.model_unet import simple_unet, actual_unet

from PIL import Image
import cv2
from keras import backend as K

dir_img = r'C:\Users\C35612\data\mid\PROSTATEx'
# Load model
model = simple_unet(96, 96)
# model = actual_unet(96, 96)
model.load_weights(r'C:\Users\C35612\data\mid\simple_unet_weights.h5')

a1 = Image.open(r'C:\Users\C35612\data\mid\my_example.jpg').convert('LA')
a1 = np.array(a1)[:,:,0]
plt.imshow(a1, cmap=plt.cm.gray)
plt.show()
b1 = skimage.transform.resize(a1, (96, 96))
b1 = b1/np.max(b1)
plt.imshow(b1, cmap=plt.cm.gray)
plt.show()
plt.hist(b1.ravel())
plt.show()
c1 = np.reshape(b1, (1,96,96,1))

# Predict with the image..
res_c1 = model.predict(c1)
res_d1 = np.reshape(res_c1, (96,96))

plt.imshow(res_d1, cmap=plt.cm.gray)
plt.show()


# Get certain images...
res_info = []
max_count = 5
count = 0
for i in os.walk(dir_img):
    if len(i[2]):
        if 't2tsetra' in i[0]:
            count += 1
            for i_file in i[2]:
                img_path = os.path.join(i[0], i_file)
                print(count, img_path)
                i_img = pydicom.read_file(img_path)
                res_info.append(i_img)
                plt.figure(count)
                plt.imshow(i_img.pixel_array, cmap=plt.cm.gray)
    if count > max_count:
        break

plt.show()

a1 = Image.open(r'C:\Users\C35612\data\mid\mirzaevinom_example.png').convert('LA')
a1 = np.array(a1)[:,:,0]
plt.imshow(a1, cmap=plt.cm.gray)
plt.show()
b1 = skimage.transform.resize(a1, (96, 96))
# b1 = b1/np.max(b1)
plt.imshow(b1, cmap=plt.cm.gray)
plt.show()
plt.hist(b1.ravel())
plt.show()
c1 = np.reshape(b1, (1,96,96,1))

# Predict with the image..
res_c1 = model.predict(c1)
res_d1 = np.reshape(res_c1, (96,96))

plt.imshow(res_d1, cmap=plt.cm.gray)
plt.show()


def pred_model(input_img, n_x=96, n_y=96):
    img_resized = skimage.transform.resize(input_img, (n_x, n_y))
    c1 = np.reshape(img_resized, (1, n_x, n_y, 1))
    res_c1 = model.predict(c1)
    res_d1 = np.reshape(res_c1, (n_x, n_y))
    return res_d1


res_img = [x.pixel_array for x in res_info]
pred_img = [pred_model(x) for x in res_img]

for i, x in enumerate(res_img):
    try:
        plt.figure(i)
        plt.imshow(x)
        plt.show()
        plt.pause(0.05)
        input()
    except KeyboardInterrupt:
        break
# Predict with the image..


plt.imshow(res_d1, cmap=plt.cm.gray)
plt.show()


a1 = Image.open(r'C:\Users\C35612\data\mid\my_example.jpg').convert('LA')
a1 = np.array(a1)[:,:,0]
plt.imshow(a1, cmap=plt.cm.gray)
plt.show()
b1 = skimage.transform.resize(a1, (96, 96))
b1 = b1/np.max(b1)
plt.imshow(b1, cmap=plt.cm.gray)
plt.show()
plt.hist(b1.ravel())
plt.show()
c1 = np.reshape(b1, (1,96,96,1))

# Predict with the image..
res_c1 = model.predict(c1)
res_d1 = np.reshape(res_c1, (96,96))

plt.imshow(res_d1, cmap=plt.cm.gray)
plt.show()

# Extract and reshape to 96x96 format
A = res_info[2].pixel_array
A0 = cv2.equalizeHist(A.astype(np.uint8))
plt.hist(A0.ravel())
plt.show()
# A1 = skimage.transform.rotate(A0, 90)
B = skimage.transform.resize(A0, (96, 96))
plt.hist(B.ravel())
plt.show()
B = skimage.transform.resize(A1, (96, 96))
# B = cv2.equalizeHist(B.astype(np.uint8))
# B1 = B/np.max(B)

derp = cv2.equalizeHist(A.astype(np.uint8))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(A.astype(np.uint8))

plt.imshow(cl1)
plt.imshow(derp)
plt.imshow(B, cmap=plt.cm.gray)
plt.imshow(A, cmap=plt.cm.gray)
plt.show()

plt.hist(B.ravel())
plt.show()
plt.hist(im_new.ravel())
plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(B1)
plt.imshow(B, cmap=plt.cm.gray)
plt.imshow(B1, cmap=plt.cm.gray)
plt.show()

C = np.reshape(B, (1,96,96,1))
C = np.reshape(B1, (1,96,96,1))

# Predict with the image..
res_C = model.predict(C)
res_D = np.reshape(res_C, (96,96))

plt.imshow(res_D, cmap=plt.cm.gray)
plt.show()


plt.hist(res_D.ravel())
plt.show()

model.predict
print()
print('Test accuracy:', score[1])

# Analyze the hidden layers of the u-net model...
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

layer_outs = functor([c1, 1.])
print(len(layer_outs))

# This is the input itself..
x1 = layer_outs[0]
print(x1)
print(x1.shape)

def fun_x(x, n=96, l=8):
    x2 = np.reshape(x1, (n, n, l))
    for i in range(l):
        plt.figure(i)
        plt.imshow(x2[:,:,i], cmap=plt.cm.gray)
        plt.show()

# This is after the first layer..
# i = 1,2 are the same...
x1 = layer_outs[1]
print(x1)
print(x1.shape)
fun_x(x1)

x1 = layer_outs[3]
print(x1.shape)
fun_x(x1, 48)

x1 = layer_outs[-2]
print(x1.shape)
fun_x(x1, 96)

x1 = layer_outs[-4]
print(x1.shape)
fun_x(x1, 96, 16)

x1 = layer_outs[-5]
print(x1.shape)
fun_x(x1, 96, 8)

x1 = layer_outs[-6]
print(x1.shape)
fun_x(x1, 96, 16)
