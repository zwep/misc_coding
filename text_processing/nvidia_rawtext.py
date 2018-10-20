# encoding: utf-8

import os
import re

path_to_text = '/home/charmmaria/data/nvidia_rawtext'

# Use an arg parse...


with open(os.path.join(path_to_text, i_file), 'r') as f:
    a_text = f.readlines()


parsed_text = [x.split('\n') for x in ' '.join(a_text).split('TIMESTAMP: ')]

overview_data = []

for i_text_stamp in parsed_text:
    # haha this is it.
    i_timestamp = i_text_stamp[0]
    raw_str_values = [x for x in i_text_stamp[1:] if '%' in x]
    str_values = [[i_timestamp, str(i_gpu).zfill(2)] + re.sub(' / ', '/', re.sub('\|', '', x)).split() for i_gpu, x in enumerate(raw_str_values)]
    overview_data.extend(str_values)

str_colname = [x for x in i_text_stamp if 'Fan' in x][0]
# We need to ditch the last one like this.. because it has a space we can fix easily.
col_names = [['Time', 'GPU'] + re.sub(' / ', '/', re.sub('\|', '', str_colname)).split()[:-1]]
col_names.extend(overview_data)
