# encoding: utf-8

"""
Showing an example with Unet on funda data

source:
https://github.com/jakeret/tf_unet/blob/master/demo/demo_toy_problem.ipynb

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util

nx = 572
ny = 572
generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)

x_test, y_test = generator(1)

fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")


net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))

path = trainer.train(generator, "./unet_trained", training_iters=20, epochs=10, display_step=2)

x_test, y_test = generator(1)

prediction = net.predict("./unet_trained/model.ckpt", x_test)
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.9
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
fig.savefig("../docs/toy_problem.png")

from PIL import Image
import numpy as np
import tensorflow as tf

file_name = 'keuken.jpg'
z = Image.open('D:/projectdata/funda/{0}'.format(file_name))
z1 = np.array(z)
min_z1 = np.min(z1)
max_z1 = np.max(z1)
# it is this easy
new_z1 = (z1 - min_z1)/(max_z1 - min_z1)

# This is normalization..
mid_ses = tf.Session()
temp_mat = tf.image.per_image_standardization(z1)
norm_img = mid_ses.run(temp_mat)

nx, ny, nz = new_z1.shape
z2 = new_z1[:, :, 0].reshape((1, nx, ny, 1))
pred = net.predict("./unet_trained/model.ckpt", z2)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(25, 25*(nx/ny)))
ax[0].imshow(z2[0,...,0], aspect="auto")
mask = pred[0,...,1] > 0.9
ax[1].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Prediction")
fig.tight_layout()
fig.savefig('D:/projectdata/funda/{0}'.format('pred_' + file_name))

