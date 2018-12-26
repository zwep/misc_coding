import config
import pandas as pd
from pandas.io import sql
import MySQLdb
from pynetgear import Netgear
import os
from datetime import datetime


seb_mysql_key = os.environ['seb_mysql_key']

con = MySQLdb.connect('localhost', 'seb', seb_mysql_key)

netgear_key = os.environ['netgear_key']
netgear = Netgear(password=netgear_key)

res_dict = []
for i in netgear.get_attached_devices():
    temp_dict = dict(zip(i._fields, list(i)))
    temp_dict['date'] = datetime.isoformat(datetime.now())
    res_dict.append(temp_dict)


A = pd.DataFrame.from_dict(res_dict)

# A.to_sql(con=con, name='table_name_for_df', if_exists='replace', flavor='mysql')
