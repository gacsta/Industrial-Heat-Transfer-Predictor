# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:30:30 2021

@author: gabr8
"""
#DATAFRAME FROM MULTIPLE LVM FILES 

import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

sinais = sorted(glob('C:/Users/gabr8/OneDrive/Documentos/Machine Learning/Sinais 05-11-2019-20210129T192834Z-001/Sinais 05-11-2019/Quadrada/quad_*.lvm'))

#Merging files into one single dataframe

Tind = pd.concat((pd.read_csv(file , encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [2]).assign(filename = file) for file in sinais), ignore_index=False, axis = 1)
Tind = Tind.drop(columns = ['filename'], axis = 0).T
Tind.reset_index(inplace = True, drop =True)

Tproc = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [1]).assign(filename = file) for file in sinais), ignore_index=False, axis = 1)
Tproc = Tproc.drop(columns = ['filename'], axis = 0).T
Tproc.reset_index(inplace = True, drop =True)

X = Tind
y = Tproc

#TEST SPLIT
X, X_test, y , y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 0)
