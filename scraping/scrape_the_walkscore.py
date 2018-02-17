import pandas as pd
import numpy as np

import lxml
from lxml import html
import requests

import re


from bs4 import BeautifulSoup

import os

loc_data = "C:\\Users\\C35612.LAUNCHER\\Testing_data\\Scraping\\Walkscore"
os.chdir(loc_data)

url_data = "https://www.walkscore.com/score/"

data_loc_sel = pd.read_csv("credab_locations_sel_scrape.csv",encoding = "latin1")

N = data_loc_sel.shape[0]
for i in np.arange(1,N):
	if i % round(N/10) == 0: print(round(i/N*100))
	i_data_loc = data_loc_sel.iloc[i]['url_loc']
	url_send = url_data + i_data_loc
	page = requests.get(url_send)
	soup = BeautifulSoup(page.content,'lxml')
	score_thing = soup.select("div.block-header-badge.score-info-link")
	if len(score_thing) >0: result = re.sub(".*(score.*png).*","\\1",str(score_thing[0]))
	else: result = ""
	data_loc_sel.loc[data_loc_sel.index[i],'walkscore'] = result
 

# Using postcodes  
os.chdir(loc_data)


url_data = "https://www.walkscore.com/score/"

data_loc_sel = pd.read_csv("WalkScore_postcode.csv",delimiter = ";",encoding = "latin1")
data_loc_sel['Huisnummer']= data_loc_sel.Huisnummer.str.strip()
data_loc_sel['Adres']  = data_loc_sel['Adres'].str.strip()
data_loc_sel['Postcode'] = data_loc_sel['Postcode'].str.lower()
data_loc_sel.Huisnummer = pd.to_numeric(data_loc_sel.Huisnummer,errors = 'coerce',downcast = 'integer')
data_loc_sel = data_loc_sel.loc[~pd.isnull(data_loc_sel.Huisnummer)]
data_loc_sel['Huisnummer']= data_loc_sel.Huisnummer.astype(int)
data_loc_sel['walkscore'] = ""

data_loc_sel_derp = data_loc_sel[['Postcode','walkscore']].drop_duplicates()

N = data_loc_sel_derp.shape[0]
for i in np.arange(0,N):
	if i % round(N/10) == 0: print(round(i/N*100))
	i_data_loc = data_loc_sel_derp.iloc[i]['Postcode']
	url_send = url_data + i_data_loc
	page = requests.get(url_send)
	soup = BeautifulSoup(page.content,'lxml')
	score_thing = soup.select("div.block-header-badge.score-info-link")
	if len(score_thing) >0: result = re.sub(".*(score.*png).*","\\1",str(score_thing[0]))
	else: result = ""
	data_loc_sel_derp.loc[data_loc_sel_derp.index[i],'walkscore'] = result
 
 
data_loc_sel = data_loc_sel.drop('walkscore',axis = 1)
data_loc_sel = data_loc_sel.merge(data_loc_sel_derp,on = 'Postcode')

data_loc_sel.walkscore = data_loc_sel.walkscore.replace("score/|\\.png","",regex = True)

os.chdir(loc_data)
data_loc_sel.to_csv("WalkScore_postcode_added.csv",index = False)
