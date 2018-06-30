# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:49:30 2018

@author: Admin
"""

import nltk

dir(nltk)
raw_data = open("SMSSpamCollection.tsv").read()
raw_data[0:500]
parsedData = raw_data.replace('\t','\n').split('\n')
parsedData[0:5]
LabelList = parsedData[0::2]