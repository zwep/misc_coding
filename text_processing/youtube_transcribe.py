# encoding: utf-8

import os
import requests
import numpy as np
from bs4 import BeautifulSoup

# to obad.. not all videos have transcription

vid_id = ['kYYJlNbV1OM', 'kYYJlNbV1OM', 'HbAZ6cFxCeY', 'wLc_MC7NQek', 'BQ4VSRg4e8w', '3iLiKMUiyTI', 'X6pbJTqv2hw', \
          'YFWLwYyrMRE', '68tFnjkIZ1Q', '4qZ3EsrKPsc', '11oBFCNeTAs', 'w84uRYq0Uc8', 'pCceO_D4AlY', 'AqkFg1pvNDw', \
          'ewU7Vb9ToXg', 'G1eHJ9DdoEA', 'D7Kn5p7TP_Y', 'fjtBDa4aSGM', 'MBWyBdUYPgk', 'Q7GKmznaqsQ', 'J9j-bVDrGdI']

dir_name = '/home/charmmaria/Music/JordanPeterson_text'

res_text = []
for i, i_vid_id in enumerate(vid_id):
    base_url = 'http://video.google.com/timedtext?lang=en&v=' + i_vid_id
    res = requests.get(base_url)
    print(i_vid_id, res.status_code)
    # bs = BeautifulSoup(res.text, 'lxml')
    # res_text.append(bs)

    file_name = 'jordan_peterson' + str(i).zfill(2) + '.html'
    file_path = os.path.join(dir_name, file_name)
    with open(file_path, "wb") as f:
      f.write(res.content)



for i, i_test in enumerate(res_text):
    file_name = 'jordan_peterson' + str(i).zfill(2) + '.txt'
    file_path = os.path.join(dir_name, file_name)
    with open(file_path, "w") as f:
      f.write(i_test.text)