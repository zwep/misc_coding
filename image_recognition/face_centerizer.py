# Extract faces en save image data as center faces in fixed size frames

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt

import numpy as np
import png
"""
locations
"""

cascPath = r'C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\code\experiment\centerFaceCapture'
imagePath = r'\\solon.prd\files\P\Global\Users\C35612\UserData\Documents\My Pictures\team'
dest_pictures = r'C:\Users\C35612.LAUNCHER\WinPython-64bit-3.5.3.1Qt5\scripts\image_team'

os.chdir(cascPath)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

"""
get faces
"""

os.chdir(imagePath)
list_images = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg")
# Read the image
"""
identify faces + cut out + save
"""

for i,i_image in enumerate(list_images):
	os.chdir(imagePath)
	image = cv2.imread(i_image)
	if image is None:
		print(i_image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_list = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(face_list) != 0:
		for j,faces in enumerate(face_list):
			#y,x = np.ogrid[int(faces[1]-round(0.3*faces[3])):int(faces[1]+faces[3]+round(0.3*faces[3])),faces[0]:(faces[0]+faces[2])]
			
			# Define circular mask
			n,m = image.shape[0:2]
			a, b = faces[1] + 0.5* faces[3],faces[0] + 0.5* faces[2]
			r = max(faces[3],faces[2])
			y,x = np.ogrid[-a:n-a, -b:m-b]
			mask = x*x + y*y <= r*r
			mask_sel = (abs(x) <= r)+ (abs(y) <= r)
			
			square_face = image[int(faces[1]-round(0.3*faces[3])):int(faces[1]+faces[3]+round(0.3*faces[3])),faces[0]:(faces[0]+faces[2]),:]
			
			image[~mask] = 255 			
			circle_face = image[int(a-r):int(a+r),int(b-r):int(b+r),:] 
			
			os.chdir(dest_pictures)
			cv2.imwrite('circle_face' + str(i) + str(j) + '.jpg',circle_face)
			cv2.imwrite('square_face' + str(i) + str(j) + '.jpg',square_face)
		#png.from_array(derp_face,'L').save('face' + str(i) + '.png')
		#plt.imshow()

 

"""
(re)scale images
"""

from resizeimage import resizeimage	
from PIL import Image

# ---- this is to mask an image with another image
# Ermegooord this works
from PIL import Image, ImageOps

os.chdir(dest_pictures)
list_pictures = glob.glob("*.jpg")


i_image = list_pictures[0]
im = Image.open(i_image)
mask = Image.open('mask.png').convert('L')
output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
output.putalpha(mask)

# ---- this is to mask an image with a self drawn image
# Now we are going to draw an Image
from PIL import Image, ImageOps, ImageDraw

size = (128, 128)
mask = Image.new('L', size, 0)
draw = ImageDraw.Draw(mask) 
draw.ellipse((0, 0) + size, fill=255)
i_image = list_pictures[0]
im = Image.open(i_image)
output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
output.putalpha(mask)
output

# ---- However, here we are just rescaling the stuff

for i_image in list_pictures:
	#fd_img = open(i_image, 'r')
	img = Image.open(i_image)
	#img = cv2.imread(i_image)
	if img.size[0] > 80:
		img = resizeimage.resize_width(img, 80)
		img.save(i_image, img.format)
	#fd_img.close()
 
