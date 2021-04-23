# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:09:30 2021

@author: Juliana
"""

#DATAFRAME FROM MULTIPLE LVM FILES 

import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np

#datascaling
from sklearn.preprocessing import MinMaxScaler

pastas = sorted(glob(r'C:\Users\Unesp\Documents\GitHub\Industrial-Heat-Transfer-Predictor\Quadrada\Quadrada 0.01\C*'))

parametrosPath = r'C:\Users\Unesp\Documents\GitHub\Industrial-Heat-Transfer-Predictor\Quadrada\Quadrada 0.01\Read-me Quadrada 0.01'

sinais = []
parametros = pd.read_csv(parametrosPath, sep="=|\s+")


m = pd.DataFrame([],  columns = ['m'])
h = pd.DataFrame([], columns = ['h'])
e = pd.DataFrame([], columns = ['e'])
A = pd.DataFrame([], columns = ['A'])
count = 0



for i in range(0,27):
    sinais = sinais + sorted(glob(pastas[i] + r'\c*_quad_*.lvm'))
    m = m.append(pd.DataFrame(np.full((102400),parametros['1'].iloc[count]), columns = ['m'] ))
    h = h.append(pd.DataFrame(np.full((102400),parametros['1'].iloc[count+2]), columns = ['h']  ))
    e = e.append(pd.DataFrame(np.full((102400),parametros['1'].iloc[count+3]), columns = ['e']  ))
    A = A.append(pd.DataFrame(np.full((102400),parametros['1'].iloc[count+1]), columns = ['A'] ))

    
    count = count+5
    


#Merging files into one single dataframe = ['m']))
   
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
m.reset_index(inplace = True, drop =True)
h.reset_index(inplace = True, drop =True)
e.reset_index(inplace = True, drop =True)
A.reset_index(inplace = True, drop =True)



X = pd.concat([Tind, Tind0, m, h, e, A], axis = 1)
X.columns = ['Tind', 'Tind0', 'm', 'h', 'e', 'A']
y = Tproc[1]





#SCALE DATA
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X.values), columns = X.columns)

#TEST SPLIT
X, X_test, y , y_test = train_test_split(X, y, train_size = 0.85, test_size = 0.15, random_state = 0, shuffle = False)

X.reset_index(inplace = True, drop =True)
y.reset_index(inplace = True, drop =True)

X_test.reset_index(inplace = True, drop =True)
y_test.reset_index(inplace = True, drop =True)

