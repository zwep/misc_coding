# dd MM yyyy format

import urllib.parse
from datetime import datetime
import numpy as np
import api_secrets
import requests
import pandas as pd
import os
import json
import pytz

from bs4 import BeautifulSoup as bs


# Import travel thing
file_name = 'reistransacties-3528010491264178.csv'
file_dir = r'C:\Users\20184098\Documents\NS Busines'
file_path = os.path.join(file_dir, file_name)

auth_obj = {'username': api_secrets.NS_username, 'x-api-key': api_secrets.NS_API_KEY}
headers_browser = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, '
                                 'like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
auth_obj.update(headers_browser)

# Prijzen-url
basis_price_url = "https://ns-api.nl/reisinfo/api/v2/price"
basis_station_url = "https://ns-api.nl/reisinfo/api/v2/stations"
basis_trips_url = "https://ns-api.nl/reisinfo/api/v3/trips"

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
        check_in_iso = str(datetime.strptime(date + check_in, "%d-%m-%y%H:%M").astimezone()).replace(' ', 'T')
        check_uit_iso = str(datetime.strptime(date + check_uit, "%d-%m-%y%H:%M").astimezone()).replace(' ', 'T')
    else:
        continue

    get_data = {'fromStation': stn1, 'toStation': stn2, 'plannedFromTime': check_in_iso}
    res = requests.get(basis_trips_url, params=get_data, headers=auth_obj)
    res.json()

    for i in dict(res.json())['trips'][0].keys():
        print(i)

    # Oke ben er bijna... nu alleen nog ff de juiste class erbij pakken... en dna klaar

    sel_travelClass = 'SECOND_CLASS'
    sel_discountType = 'NO_DISCOUNT'
    sel_discountType = 'DISCOUNT_40_PERCENT'
    sel_product = 'OVCHIPKAART_ENKELE_REIS'

for i in dict(res.json())['trips'][0]['fares']:
    print(i)