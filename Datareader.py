# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:19:17 2018

@author: Admin
"""

from pandas_datareader import data, wb
import pandas as pd
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2016,7,15)
yahoo_df = data.DataReader("F",'google',start,end)
yahoo_df.plot()
