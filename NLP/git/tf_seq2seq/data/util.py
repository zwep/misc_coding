#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Utility functions

This code is based on the util.py of 
nematus project (https://github.com/rsennrich/nematus)
"""

import sys
import json
# import cPickle as pkl  cPickle is only for Python 2.
import pickle as pkl

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion


def unicode_to_utf8(d):
    """

    :param d:
    :return:
    """
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    """

    :param filename:
    :return:
    """
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    """

    :param basename:
    :return:
    """
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)
