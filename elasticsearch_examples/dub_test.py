

# taken from https://tryolabs.com/blog/2015/02/17/python-elasticsearch-first-steps/

import requests
from elasticsearch import Elasticsearch
import json

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import MultiMatch

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch([{'host': '192.168.1.11', 'port': 9200}])
es.indices.get_alias("*")
es.search(index='news')
es.search(index='dutch_news')

s = Search(using=es)
s_match = s.query("match", category="sport")

def some_test(input_query, i_field):
    s_fuzzy = s.query("multi_match", query=input_query, fields=i_field)
    res = s_fuzzy.execute()
    for i, i_item in enumerate(res):
        print(i, i_item.date, i_item.content, "\n\n")


some_test('Melissa', ['content', 'date'])
s_match[:10].execute()
res = s.execute()
for i in res:
    print(i)


s = Search(using=es, index="dutch_news")
max_count = s.count()
res = s[0:max_count].execute()
res.hits.total
len(res.hits.hits)
from dateutil.parser import parse as duparse
res_time = [duparse(x['_source']['date']) for x in res.hits.hits]
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.array(res_time))
plt.hist(res_time)

import collections
for i in collections.Counter([x.hour for x in res_time]).items():
    print(i)

z = np.array(list(collections.Counter([x.hour for x in res_time]).values()))
z = np.array(list(collections.Counter([x.day for x in res_time]).values()))
z = np.array(list(collections.Counter([x.month for x in res_time]).values()))

plt.plot(z)

# Here we are going to check netgear results


# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch([{'host': '192.168.1.11', 'port': 9200}])
es.indices.get_alias("*")
len(es.search(index='netgear'))


s = Search(using=es, index="netgear")
max_count = s.count()
res = s[0:max_count].execute()
res[2]
z = res.hits.hits[0]
import dateutil.parser
res_parsed = [(dateutil.parser.parse(x['_source']['date']), x['_source']['ip'], x['_source']['name'], x['_source'][
    'signal']) for x in res.hits.hits if 'ip' in x['_source'].keys()]


import pandas as pd
import datetime
A = pd.DataFrame(res_parsed)
A.columns = ['date', 'ip', 'name', 'signal']
A['datehour'] = A['date'].dt.to_period('H')
A['datehour2'] = A['datehour'].map(lambda x: x.to_timestamp())

# Not sure how to group by two.. and plot a third...
A.groupby(['datehour2', 'name']).size().unstack().plot(kind='bar', stacked=True)

