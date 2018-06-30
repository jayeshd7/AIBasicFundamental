# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:05:18 2018

@author: Admin
"""
import pandas as pd
file_name_string = 'C:/Users/Admin/Desktop/Stats/Ex_Files_Pandas_Data/Ex_Files_Pandas_Data/Exercise Files/02_07/Final/EmployeesWithGrades.xlsx'

employee_df = pd.read_excel(file_name_string,'Sheet1',index_col = None,na_values =['NA'])
employee_df
employee_df.groupby('Department').sum()