# encoding: utf-8

"""

In this piece we create two classes

- InitLabels
- EditLabels

The former one is to initiate some predefined labels that I have found myself to be useful.
The latter one is used to edit these to some personal preference.
Or just add more information to it.

"""

import collections
import json


class InitLabels:
    """
    Hallo hier maken we wat labels
    """
    def __init__(self):
        self.label_dict = self.get_label_dict()

    def get_label_dict(self):
        tmp_dict = collections.defaultdict(dict)
        tmp_dict['city'].update(self._init_city_label())
        tmp_dict['shop'].update(self._init_shop_label())
        tmp_dict['supermarkt'].update(self._init_supermarkt_label())
        return tmp_dict

    def write_json(self, name_json='label_dict.txt', dir_json=''):
        """
        Writes the dict with labels to a json file
        :param name_json:
        :param dir_json:
        :return:
        """
        with open(dir_json + name_json, 'w') as f:
            json.dump(self.label_dict, f, ensure_ascii=False)

    @staticmethod
    def _init_type_label():
        """

        :return:
        """
        name_abbr = ['ac', 'ba', 'bc', 'bg', 'cb', 'ck', 'db', 'ei', 'ga', 'gb', 'id', 'kh', 'ma', 'nb', 'sb', 'sp',
                     'tb', 'tg']

        name_long = ['acceptgiro', 'betaalautomaat', 'betalen contactloos', 'bankgiro_opdracht',
                     'crediteurenbetaling', 'Chipknip', 'diverse_boekingen', 'euro-incasso', 'geldautomaat_Euro',
                     'geldautomaat_VV', 'iDEAL', 'kashandeling', 'machtiging', 'NotaBox', 'salaris_betaling',
                     'spoedopdracht', 'eigen_rekening', 'telegiro']

        return dict(zip(name_abbr, name_long))

    @staticmethod
    def _init_supermarkt_label():
        """
        :return: dictionary with query and labels of supermarkets
        """

        query_supermarkt = ['Agrimarkt', 'Albert Heijn', 'AH to go', 'Aldi', 'Attent', 'Attent Super op vakantie!',
                            'Boni', 'Coop', 'CoopCompact', 'Dagwinkel', 'Deen', 'Deka Markt', 'Dirk',
                            'EMTÉ Supermarkten', 'E-markt', 'Hoogvliet', 'Jan Linders', 'Jumbo', 'Kingsalmarkt',
                            'Lidl', 'MCD', 'Makro', 'M&M supermarkten', 'Nettorama', 'Pakgro', 'Picnic', 'Plus',
                            'Poiesz', 'Pryma', 'Sligro', 'Spar', 'Supercoop', 'Troefmarkt', 'Vomar', 'Boon',
                            'Albertheijn']

        query_supermarkt = [x.upper() for x in query_supermarkt]

        label_supermarkt = ['Agrimarkt', 'AH', 'AH', 'Aldi', 'Attent', 'Attent', 'Boni',
                            'Coop', 'Coop', 'Dagwinkel', 'Deen', 'Deka Markt', 'Dirk', 'EMTÉ Supermarkten',
                            'E-markt', 'Hoogvliet', 'Jan Linders', 'Jumbo', 'Kingsalmarkt', 'Lidl', 'MCD', 'Makro',
                            'M&M supermarkten', 'Nettorama', 'Pakgro', 'Picnic', 'Plus', 'Poiesz', 'Pryma', 'Sligro',
                            'Spar', 'Supercoop', 'Troefmarkt', 'Vomar', 'Boon', 'AH']

        return dict(zip(query_supermarkt, label_supermarkt))

    @staticmethod
    def _init_shop_label():
        """
        :return: dictionary with query and labels of shops
        """
        query_shop = ['Mediamarkt', 'Decathlon', 'Gamma', 'Ikea', 'Hema', 'Kruidvat', 'Gall.+Gall']

        query_shop = [x.upper() for x in query_shop]

        label_shop = ['Mediamarkt', 'Decathlon', 'Gamma', 'Ikea', 'Hema', 'Kruidvat', 'Gall.+Gall']

        return dict(zip(query_shop, label_shop))

    @staticmethod
    def _init_city_label():
        """
        :return: dictionary with query and labels of cities
        """
        query_city = ['Delft',  'Utrecht',  'Zaandam',  'Amsterdam',  'Amsterdam Zui',  'Rotterdam',  "'s-Gravenhage",
                      'Voorburg',  'Schiphol',  'Capelle ad ys',  'Arnhem',  'Leiden',  'Amersfoort',  'Ede',
                      'Hertogenbo', 'Muiden']

        query_city = [x.upper() for x in query_city]

        label_city = ['Delft',  'Utrecht',  'Zaandam',  'Amsterdam',  'Amsterdam Zuid',  'Rotterdam',  "Den Haag",
                      'Voorburg',  'Schiphol',  'Capelle aan den Ijssel',  'Arnhem',  'Leiden',  'Amersfoort',  'Ede',
                      'Den Bosch', 'Muiden']
        return dict(zip(query_city, label_city))


class EditLabels:
    """
    This class allows us to edit a json file with labels...
    """
    def __init__(self, file_name, path=''):
        """

        :param file_name:
        :param path:
        """
        self.label_dict = self.load_json(file_name, path)
        self.file_name = file_name
        self.path = path

    @staticmethod
    def load_json(file_name, path):
        """
        Used to load the labels
        :return:
        """
        with open(path + file_name) as f:
            loaded_json = json.load(f)
        return loaded_json

    def write_json(self):
        """
        Writes the dict with labels to a json file
        :return:
        """
        with open(self.path + self.file_name, 'w') as f:
            json.dump(self.label_dict, f, ensure_ascii=False)

    def get_cat(self):
        """
        Convenience function to retrieve all categories
        :return:
        """
        return list(self.label_dict.keys())

    def get_label(self, cat):
        """
        Convenience function to retrieve all labels of a given category
        :param cat:
        :return:
        """
        return list(self.label_dict[cat].values())

    def get_query(self, cat):
        """
        Convenience function to retrieve all queries for a given category
        :param cat:
        :return:
        """
        return list(self.label_dict[cat].keys())

    def update_label(self, update_dict, cat):
        """
        Update provided labels of a given category
        :param update_dict:
        :param cat:
        :return:
        """
        self.label_dict[cat].update(update_dict)

    def update_cat(self, update_dict):
        """

        :param update_dict:
        :return:
        """
        self.label_dict.update(update_dict)

    def remove_query(self, cat, pop_keys):
        """

        :param cat:
        :param pop_keys:
        :return:
        """
        return [self.label_dict[cat].pop(i_pop) for i_pop in pop_keys]

    def remove_cat(self, pop_keys):
        """

        :param pop_keys:
        :return:
        """
        return [self.label_dict.pop(i_pop) for i_pop in pop_keys]
