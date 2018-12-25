# encoding: utf-8

"""

"""

import os
import cv2
cascade_path = '/home/charmmaria/Downloads/opencv-4.0.0-beta/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

image_dir = '/home/charmmaria/Pictures/TOSort'
face_img_dir = '/home/charmmaria/Pictures/TOSort_faces'
no_face_img_dir = '/home/charmmaria/Pictures/TOSort_nofaces'


z_ext = [os.path.splitext(x)[1] for x in os.listdir(image_dir)]
import collections
collections.Counter(z_ext)

list_images = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if os.path.splitext(x)[1] in ['.jpg',
                                                                                                       '.JPG',
                                                                                                       '.jpeg',
                                                                                                       '.png']]
sel_list_images = list_images[0:100]

result_dict = {}
result_dict['no_face'] = []
result_dict['face'] = []


for i,i_image in enumerate(sel_list_images):
    face_counter = 0
    image = cv2.imread(i_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_list = face_cascade.detectMultiScale(gray, 1.5, 5)
    if len(face_list):
        store_face = face_list
        store_img = image
        result_dict['face'].append(i_image)
        for i_face in face_list:
            i_file_name, i_ext = os.path.splitext(i_image)
            i_file_name = os.path.basename(i_file_name)

            y1 = (i_face[0] + i_face[2])
            x1 = (i_face[1] + i_face[3])
            i_face_img = store_img[i_face[1]:x1, i_face[0]:y1]
            i_face_name = i_file_name + '_face_' + str(face_counter).zfill(3) + '.png'
#            print(i_face_img.shape, i_face, image.shape)
            cv2.imwrite(os.path.join(face_img_dir, i_face_name), i_face_img)
            face_counter+=1
    else:
        cv2.imwrite(os.path.join(no_face_img_dir, os.path.basename(i_image)), image)
        result_dict['no_face'].append(i_image)

