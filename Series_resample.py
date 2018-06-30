# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:47:50 2018

@author: Admin
"""

import pandas as pd
import numpy as np
my_index = pd.date_range('1/4/2018',periods=9,freq='min')
my_index

my_series = pd.Series(np.arange(9),index= my_index)
my_series.resample('3min',label='right').sum()
my_series.resample('30S').asfreq()[0:15]