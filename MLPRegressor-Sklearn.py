# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:06:34 2021

@author: gabr8

"""

import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


#GET_DATA_FROM

from Tind_Tindn_Gama import *

# from Sem_Ruido_Basico_sem_Parametros import *
# X = Tind
# y = Tproc

# mat = scipy.io.loadmat('C:/Users/gabr8/Downloads/ML stuff/ML stuff/Recesso.Arquivos-20200321T191135Z-001/Recesso.Arquivos/Temp_13-09-2019 ruido 002.mat')

# Inputs = np.concatenate((mat['TempSenoInput'].T, mat['TempQuadInput'].T, mat['TempSerInput'].T, mat['TempPadInput'].T, mat['TempTriInput'].T), axis = 0)
# Outputs = np.concatenate((mat['TempSenoOutput'].T, mat['TempQuadOutput'].T, mat['TempSerOutput'].T, mat['TempPadOutput'].T, mat['TempTriOutput'].T), axis = 0)

# X = pd.DataFrame(data = Inputs)
# y = pd.DataFrame(data = Outputs)


def model_tester(X, y, hl):      
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 0)
    
    
    regr = MLPRegressor(random_state=0, max_iter=100000, hidden_layer_sizes = hl).fit(X_train, y_train)
    Net = regr.predict(X_test[:])
    Error = mean_absolute_error(y_test, Net)
    
   
    return Net, X_test, y_test, Error, hl

#hidden layer sizes
#500 --> 2000

hl = np.arange(10, 100, 10)
modelos = []

for i in hl:
    for p in hl:
        Net = model_tester(X, y, (i,p,))[0]
        Tind = model_tester(X, y, (i,p,))[1]
        Tproc = model_tester(X, y, (i,p,))[2]
        Error = model_tester(X, y, (i,p,))[3]
        
        modelos.append(model_tester(X, y, (i,p)))
        
        if Error < 0.4:
            fig = plt.figure()
            Tind.iloc[0, :].plot(color = 'r', label = 'Tind')
            Tproc.iloc[17, :].plot(color = 'b', label = 'Tproc')
            plt.plot(np.arange(0,1024), Net[0,:], 'g', label = 'Net')
            title = 'Hidden Layer Sizes --> {}  {}  Error --> {}'.format(i, p,  Error)
            fig.suptitle(title)
   






