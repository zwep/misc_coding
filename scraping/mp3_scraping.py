import requests
from bs4 import BeautifulSoup as bs
import re
import os
import time
import numpy as np

storage_path = r'C:\Users\20184098\Music'

n = 1
n_max = 11
no_name_counter = 1
for n in range(1,n_max):
    n_page = 'https://www.bbc.co.uk/programmes/p01dh5yg/episodes/downloads?page={}'.format(str(n))
    doc = requests.get(n_page)
    doc_bs = bs(doc.text, 'lxml')
    doc_href_res = doc_bs.find_all(href=True)

    re_download = re.compile('https.*download/proto.*mp3')
    re_mp3name = re.compile(', ([A-Z].*) -')


    for x in doc_href_res:
        if re_download.match(x['href']):
            re_group = re_mp3name.findall(x['download'])
            if len(re_group):
                name_mp3_file = ''.join(re_group[0].split())
            else:
                name_mp3_file = 'NoName{}'.format(no_name_counter)
                no_name_counter += 1

            print('Getting {x}, with url {y}'.format(x=name_mp3_file, y=x['href']))
            doc_mp3 = requests.get(x['href'])
            with open(storage_path + '\\' + name_mp3_file + '.mp3', 'wb') as f:
                f.write(doc_mp3.content)
            print('\t Stored to ' + storage_path + '\\' + name_mp3_file + '.mp3')
            # Just wait for a little while...
            time.sleep(np.random.randint(5,20))  # seconds