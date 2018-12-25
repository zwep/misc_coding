# encoding: utf-8

"""

This script allows you to scan a directory (with images) and move these images to another directory, according to
the amount of faces that can be found in the images.

The use case for this is that nowadays we have tons and tons of pictures... but we dont have the time to sort them all.
This comes in use when you want to process your vacation pics, or just a dump of images.

Later, this can be improved to search for specific faces, or maybe for specific scenes/locations.
For now, it just sort the face images to either

	* directory that contains no faces (0)
	* directory that contains one face (1)
	* directory that contains many faces (>1)
	
Specifying different values is of course possible, but with this, one only needs to create three extra directories.

"""

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt

import numpy as np
import png

import shutil

"""
locations
"""

# This is the path to the cascade.html
# Can be downloaded from : 
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
#
cascPath = r'C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\code\experiment_settoIdeaProject\centerFaceCapture'
imagePath = r'\\solon.prd\files\P\Global\Users\C35612\UserData\Documents\My Pictures\team'
destPath = r'C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\code\experiment_settoIdeaProject\centerFaceCapture'

face_cascade = cv2.CascadeClassifier(cascPath + '\haarcascade_frontalface_default.xml')
list_images = glob.glob(imagePath + "\*.jpg") + glob.glob(imagePath + "\*.png") + glob.glob(imagePath + "\*.jpeg")

# Read the image
"""
identify faces + cut out + save
"""

number_of_faces = []
for i, i_image in enumerate(list_images):

	base_name = os.path.basename(i_image)
	os.chdir(imagePath)
	image = cv2.imread(i_image)
	
	
	if image is None:  # Make sure that we actually have an image
		print(i_image)
		break
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_list = face_cascade.detectMultiScale(gray, 1.3, 5)  # Determine nr of faces
	
	if len(face_list) != 0:
		N_face = len(face_list)
		N_face_tuple = (i_image, N_face)
		number_of_faces.append(N_face_tuple)
		
		# ugly ugly if-statements, but it is readable.
		if N_face == 0:
			# os.rename(i_image, destPath + '\no_faces\\' + base_name)
			shutil.move(i_image, destPath + '\no_faces\\' + base_name)
		elif N_face == 1:
			# os.rename(i_image, destPath + '\one_face\\' + base_name)
			shutil.move(i_image, destPath + '\one_face\\' + base_name)
		elif N_face > 1:
			# os.rename(i_image, destPath + '\many_faces\\' + base_name)
			shutil.move(i_image, destPath + '\many_faces\\' + base_name)
		else:
			# os.rename(i_image, destPath + '\other\\' + base_name)	
			shutil.move(i_image, destPath + '\other\\' + base_name)
			