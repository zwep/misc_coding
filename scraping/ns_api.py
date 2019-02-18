# dd MM yyyy format

from datetime import datetime
import numpy as np
import api_secrets
import requests
import pandas as pd
import os
import json
from bs4 import BeautifulSoup as bs


# Import travel thing
file_name = 'reistransacties-3528010491264178.csv'
file_dir = r'C:\Users\20184098\Documents\NS Busines'
file_path = os.path.join(file_dir, file_name)

auth_obj = {'username': api_secrets.NS_username, 'x-api-key': api_secrets.NS_API_KEY}

# Prijzen-url
basis_price_url = "https://ns-api.nl/reisinfo/api/v2/price"
basis_station_url = "https://ns-api.nl/reisinfo/api/v2/stations"

# Station lijst
json_station = json.loads(requests.get(basis_station_url, headers=auth_obj).text)
dict_station = dict(json_station)['payload']
station_code = {x['namen']['lang']: x['code'] for x in dict_station}
via_ind = False

# Eigen reizen
travel_list = pd.read_csv(file_path, encoding='latin', sep=';')

for i, i_row in travel_list.iterrows():
    if i_row['Vertrek'] is not np.nan:
        stn1 = i_row['Vertrek']
        stn2 = i_row['Bestemming']

        date = i_row['Datum']
        check_in = i_row['Check in']
        check_uit = i_row['Check uit']
        check_in_iso = str(datetime.strptime(date + check_in, "%d-%m-%y%H:%M"))
        check_uit_iso = str(datetime.strptime(date + check_uit, "%d-%m-%y%H:%M"))
    else:
        continue

    travel_url = "?fromStation={stn1}&toStation={stn2}&plannedFromTime={checkin}".format(stn1=station_code[stn1],
                                                                                    stn2=station_code[stn2],
                                                                                    checkin=check_in_iso[
                                                                                            :-3].replace(' ', 'T'))
    price_url = basis_price_url + travel_url

    res = requests.get(price_url, headers=auth_obj)
