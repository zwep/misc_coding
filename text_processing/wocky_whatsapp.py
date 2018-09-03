import collections
import itertools
import os
import re

location_data = '/home/charmmaria/data/Wocky'
name_files = os.listdir(location_data)

res_text = []
for i_file in name_files:
    i_file_path = os.path.join(location_data, i_file)
    with open(i_file_path , 'r') as f:
        i_text = f.readlines()
        res_text.append((i_file, i_text))

last_item = [(a, b[-1][0:10]) for a, b in res_text]
for i in last_item:
    print(i)

# Do something fancy to detect the most recent text file...
id_last_text = 3
recent_text = res_text[id_last_text][1]

# See when she said something...
search_message_sophie = [(i, x) for i, x in enumerate(recent_text) if 'Sophie Wocky' in x]

# Found these by inspection..
id_second_last = 8515
id_last = 8829
date_second_last = '24/08/2018'
date_last = '03/09/2018'

# Now find all the messages between these dates..
id_seq_sl = min([i for i, x in enumerate(recent_text) if date_second_last in x])
id_seq_l = max([i for i, x in enumerate(recent_text) if date_last in x])

# Choose to use the self-found last id...
last_300 = recent_text[id_seq_sl:(id_last+1)]
garbage_regex = '([0-9]{2}/[0-9]{2}/[0-9]{4}, [0-9]{2}:[0-9]{2} - )'
clean_text_names = [re.findall(garbage_regex + '(.*)', x)[0][1] for i, x in enumerate(last_300) if re.match(
    garbage_regex + '(.*)', x)]

clean_text = [re.findall('(.*?): (.*)', x)[0][1] for x in clean_text_names]
name_occurence = [re.findall('(.*?): (.*)', x)[0][0] for x in clean_text_names]

unique_names = list(set(name_occurence))
aantal_mensen = len(unique_names)

text_name_tuple = list(zip(name_occurence, clean_text))
length_text_name_tuple = [(x, len(y)) for x,y in text_name_tuple]

avg_sent_length = {}
for i_name in unique_names:
    sent_name = [y for x,y in length_text_name_tuple if i_name == x]
    avg_sent_length[i_name] = {'count': len(sent_name),
                               'avg_length': round(sum(sent_name)/len(sent_name))}

count_distr = [v['count'] for k, v in avg_sent_length.items()]
max_count_distr = max(count_distr)
id_max_count = count_distr.index(max_count_distr)
person_max_count = list(avg_sent_length.keys())[id_max_count]

avglength = [v['avg_length'] for k, v in avg_sent_length.items()]
max_avglength_distr = max(avglength )
id_max_avglength = avglength .index(max_avglength_distr )
person_max_avglength = list(avg_sent_length.keys())[id_max_avglength]

name_distr = collections.Counter(name_occurence)

full_text = '. '.join([x for x in clean_text if not x == '<Media omitted>'])

from summarizer import summarize
summarize('Wocky', full_text, 10)
# Hoe vaak mensen iets zeiden
# Wat van alle text het meest voorkomende woord is
# Wat een samenvatting geeft wat alles
