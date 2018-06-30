# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:10:01 2018

@author: Jayesh
"""

import pandas as pd

data = pd.read_csv("SMSSpamCollection.tsv",sep ='\t')
data.columns = ['label','body_text']
data['body_len']= data['body_text'].apply(lambda x:len(x)-x.count(" "))
data.head()