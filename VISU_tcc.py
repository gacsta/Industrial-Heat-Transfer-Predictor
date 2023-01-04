# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:33:28 2022

@author: gabr8
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
# import scipy.fftpack
import pandas as pd
# num = 40
df2 = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\Seno2808_teste.csv')
df2 = df2.iloc[:, 1:]

df3 = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Seno_reg_teste.csv')



# In[ ]:

#tind + h
X_test = pd.concat([df2.iloc[:,0:128], df2.iloc[:,129]], axis = 1)
Trec = df2['Trec']
y_test = df2.iloc[:,128]

num = 15

plt.figure(figsize=(12, 7))
plt.plot(np.arange(0,896), Trec[(num*896):((num*896)+896)], 'y', label = 'Trec')
plt.plot(np.arange(0,896),y_test[(num*896):((num*896)+896)], 'b', label = 'Tproc')
# plt.plot(np.arange(0,896),y_pred[(num*896):((num*896)+896)], 'g')
plt.plot(np.arange(0,896),X_test['2'].loc[(num*896):((num*896)+895)], 'r', label = 'Tind')
plt.legend()


