# encoding: utf-8

import requests
from bs4 import BeautifulSoup as bs


url = 'https://www.google.co.il/search?q=eminem+twitter'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'

# header variable
headers = { 'User-Agent' : user_agent }


url_interveste_huur = 'https://www.interveste.nl/woonruimte/tijdelijk-huren/'

n_pp = 9
max_count = 100
all_is_good = True
webpage = 1

house_list = []
while all_is_good and webpage < max_count:
    webpage += 1
    url_page = url_interveste_huur + str(webpage * n_pp)
    res = requests.get(url_page)
    res = urllib.request.urlopen(url_page)
    if res.status_code == 200:
        res_text = bs(res.text, 'lxml')
        link_to_house = res_text.findAll('a'), {'class': 'aanbod-link'})
        house_list.extend(link_to_house)
    else:
        print('\nError: ', res.status_code)
        print('\nPage: ', webpage)
