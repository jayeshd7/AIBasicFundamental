# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:11:21 2018

@author: Admin
"""

import pandas as pd
import numpy as np
startingdate = '20180701'
sample_numpy_data = np.array(np.arange(24).reshape(6,4))
dates_index = pd.date_range(startingdate,periods=6)
sample_df = pd.DataFrame(sample_numpy_data,index= dates_index,columns=list('ABCD'))
sample_df
sample_df_2 = sample_df.copy()
sample_df_2['Fruits'] = ['apple','mango','pinaple','banana','watermellon','chiku']
sample_df_2
ExtraData = pd.Series([1,2,3,4,5,6],index= pd.date_range(startingdate,periods=6))
sample_df_2['ExtraData']=ExtraData*3+1
sample_df_2
second_numpy_array = np.array(np.arange(len(sample_df_2)))*100 +7
sample_df_2['G'] = second_numpy_array

pd.set_option('display.precision',2)
sample_df_2.mean(1)
pieces = [sample_df_2[:2],sample_df_2[2:4],sample_df_2[4:]]
pieces[1]
newlist = pieces[0],pieces[1]
pd.concat(newlist)
sample_last_row = sample_df_2.iloc[2]
sample_df_2.append(sample_last_row)



