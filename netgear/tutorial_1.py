# encoding: utf-8

"""
To obtain netgear data...
"""

from pynetgear import Netgear
import config
import os
from datetime import datetime
import pandas as pd


def get_netgear_devices():
    netgear_key = os.environ['netgear_key']
    netgear = Netgear(password=netgear_key)

    res_dict = []
    for i in netgear.get_attached_devices():
        temp_dict = dict(zip(i._fields, list(i)))
        temp_dict['date'] = datetime.isoformat(datetime.now())
        res_dict.append(temp_dict)

    return pd.DataFrame.from_dict(res_dict)
