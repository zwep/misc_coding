# encoding: utf-8

"""
Here we are trying several ways to obtain information from the BAG

Because apparantly there are multiple sources on how to get BAG data

-> overheid.io : https://overheid.io/documentatie/bag
-> PDOK : https://data.pdok.nl/bag/api/v1
-> nationaalgeoregister?

"""

import requests

# This is my personal API key
request_header = {'X-Api-Key': 'get your own api key'}

# Using BAG.BASISREGISTRATIES ----------------------------------
# Getting some postcode..
bag_num_url = 'https://bag.basisregistraties.overheid.nl/api/v1/nummeraanduidingen'
bag_postcode_request = '?postcode=2628EX'
bag_num_url_req = bag_num_url + bag_postcode_request
req_num = requests.get(bag_num_url_req, headers=request_header)
print(req_num.status_code)

# Getting some bag-id..
bag_vbo_url = 'https:/bag.basisregistraties.overheid.nl/api/v1/verblijfsobjecten'
bag_vboid_request = '/0879010000006009'
bag_vboid_url_req = bag_vbo_url + bag_vboid_request
req_vbo = requests.get(bag_vboid_url_req, headers=request_header)
print(req_vbo.status_code)

# Using GEODATA ----------------------------------
# Here, I believe, you can also do something like 'suggest?q=Utrecht' to get all the objects in Utrecht.
# Somehow, you don't need an API key for this server, quite weird.
bag_url = 'https://geodata.nationaalgeoregister.nl/locatieserver/v3/suggest?q=2628EX'
req_bag = requests.get(bag_url)
print(req_bag.status_code)

# Using OVERHEID.IO ----------------------------------
# Here yoeu h
overheid_header = {'ovio-api-key': 'e7b3605b3fbffaaf31bb16f87292aaf2911a8cd242c037326f03e08b2df8d260'}
overheid_url = 'https://overheid.io/api/bag?filters[postcode]=3015BA'
req_overheid = requests.get(overheid_url, headers=overheid_header)
print(req_overheid.status_code)
