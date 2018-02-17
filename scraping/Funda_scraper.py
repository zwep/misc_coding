
# Create a robot as well to inform their server

# Scraping funda posses some problems! Quite funny actually

import lxml
from bs4 import BeautifulSoup as BS
import numpy as np
import requests
import cssselect

import lxml.html

url_link = 'https://www.funda.nl/huur/heel-nederland/'
response = requests.get(url_link)
tree = lxml.html.fromstring(response.text)
title_elem = tree.xpath('.search-result-title')
title_elem = tree.cssselect('.search-result-title')[0]  # equivalent to previous XPath
print("title tag:", title_elem.tag)


url_link = 'https://www.funda.nl/huur/lent/huis-40407182-dalidastraat-49/'
response = requests.get(url_link)
tree = lxml.html.fromstring(response.text)

bs = BS(response.text, 'lxml')

attrs = []
for elm in soup():  # soup() is equivalent to soup.find_all()
    attrs += list(elm.attrs.values())

print(attrs)


# title_elem = tree.xpath('//title')[0]
title_elem = tree.cssselect('.search-result-title')[0]  # equivalent to previous XPath
print("title tag:", title_elem.tag)

id
'gtmDataLayer'

# --- this looks pretty cool
# http://docs.python-requests.org/en/latest/index.html
s = requests.Session()
s.auth = ('user', 'pass')
s.headers.update({'x-test': 'true'})

# both 'x-test' and 'x-test2' are sent
s.get('http://httpbin.org/headers', headers={'x-test2': 'true'})



UAS = ("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1", 
       "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0",
       "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
       )

ua = UAS[0]

headers = {'user-agent': ua}
response = requests.get(url_link, headers = headers)