# coding: utf-8

"""
Created on Mon May 14 12:48:18 2018
@author: Jurjen Meerman

Video to Gif
"""

import numpy as np
import cv2
from PIL import ImageGrab
import ctypes

# Get screen resolution (for full screen capture)
user32 = ctypes.windll.user32
screensize_x, screensize_y = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
# Exit key
esc_key = 27


bbox_vid = (0, 82, screensize_x, screensize_y-50-82)  # x, y, w, h
# This makes sure that what we are writing is the same size as what we are recording
bbox_writer = (bbox_vid[2] - bbox_vid[0], bbox_vid[3] - bbox_vid[1])

# Define video
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
vid = cv2.VideoWriter('record_test.avi', fourcc, 40, bbox_writer)

# Run the video grabbing thing..
while(True):
    img = ImageGrab.grab(bbox=bbox_vid)
    img_np = np.array(img)
    RGB_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    vid.write(RGB_img)

    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == esc_key:
        break

vid.release()
cv2.destroyAllWindows()
