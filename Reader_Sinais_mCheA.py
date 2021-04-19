# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:09:30 2021

@author: Juliana
"""

#DATAFRAME FROM MULTIPLE LVM FILES 

import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

#datascaling
from sklearn.preprocessing import MinMaxScaler

pastas = sorted(glob(r'C:\Users\Juliana\Documents\Gabriel A. N. S. Costa\Github\Industrial-Heat-Transfer-Predictor\Sinais 05-11-2019-20210129T210817Z-001\Quadrada-20210416T173822Z-001\Quadrada\Quadrada 0.01\C*'))

sinais = []
for i in range(0,27):
    sinais = sinais + sorted(glob(pastas[i] + r'\c*_quad_*.lvm'))


#Merging files into one single dataframe

Tind = pd.concat((pd.read_csv(file , encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [2]).assign(filename = file) for file in sinais), ignore_index=False, axis = 0)
Tind = Tind.drop(columns = ['filename'], axis = 0)

Tproc = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [1]).assign(filename = file) for file in sinais), ignore_index=False, axis = 0)
Tproc = Tproc.drop(columns = ['filename'], axis = 0)

Gama = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [6]).assign(filename = file) for file in sinais), ignore_index=False, axis = 0)
Gama = Gama.drop(columns = ['filename'], axis = 0)

Trec = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [3]).assign(filename = file) for file in sinais), ignore_index=False, axis = 0)
Trec = Trec.drop(columns = ['filename'], axis = 0)

Tind0 = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col= None, header = None, usecols = [4]).assign(filename = file) for file in sinais), ignore_index=False, axis = 0)
Tind0 = Tind0.drop(columns = ['filename'], axis = 0)



Tind.reset_index(inplace = True, drop =True)
Tproc.reset_index(inplace = True, drop =True)
Gama.reset_index(inplace = True, drop =True)
Trec.reset_index(inplace = True, drop =True)
Tind0.reset_index(inplace = True, drop =True)


X = pd.concat([Tind, Tind0], axis = 1)
X.columns = ['Tind', 'Tind0']

y = Tproc



# #SCALE DATA
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

# #TEST SPLIT
# X, X_test, y , y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 0)
