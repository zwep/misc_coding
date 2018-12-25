# encoding: utf-8

import numpy as np
import os
import pynetgear


print('Starting script\n')
from pynetgear import Netgear
from elasticsearch import Elasticsearch
import os
from datetime import datetime
print('Loaded all libraries\n')


netgear = Netgear(password=netgear_key)

res_dict = []
for i in netgear.get_attached_devices():
    temp_dict = dict(zip(i._fields, list(i)))
    temp_dict['date'] = datetime.isoformat(datetime.now())
    res_dict.append(temp_dict)

res_dict[0]
