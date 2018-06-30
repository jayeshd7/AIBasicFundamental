# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:55:35 2018

@author: Admin
"""

import pandas as pd
import numpy as np

browser_index = ['Firefox','chrome','safari','ie','uc']
browser_df = pd.DataFrame({
        'httpStatus' : [200,200,404,203,201],
        'responcetime' : [0.04,0.02,0.01,0.05,0.05]
        },index= browser_index)
browser_df


new_index = ['chrome','safari','uc','firefox','ie']
browser_df_2 = browser_df.reindex(new_index)
browser_df_2
browser_df_3 = browser_df_2.dropna(how ='any')
browser_df_4 = browser_df_2.fillna(value = 0.005)