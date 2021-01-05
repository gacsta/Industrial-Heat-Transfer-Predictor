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


mat = scipy.io.loadmat('C:/Users/gabr8/Downloads/ML stuff/ML stuff/Recesso.Arquivos-20200321T191135Z-001/Recesso.Arquivos/Temp_09-09-2019 ruido 002.mat')

Inputs = np.concatenate((mat['TempSenoInput'].T, mat['TempQuadInput'].T, mat['TempSerInput'].T, mat['TempPadInput'].T, mat['TempTriInput'].T), axis = 0)
Outputs = np.concatenate((mat['TempSenoOutput'].T, mat['TempQuadOutput'].T, mat['TempSerOutput'].T, mat['TempPadOutput'].T, mat['TempTriOutput'].T), axis = 0)

X = pd.DataFrame(data = Inputs)
y = pd.DataFrame(data = Outputs)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)



regr = MLPRegressor(random_state=1, max_iter=100000, hidden_layer_sizes = (600)).fit(X_train, y_train)
Net = regr.predict(X_test[:])
score = regr.score(X_test, y_test)

plt.figure()
X_test.iloc[1, :].plot()
X_train.iloc[1, :].plot()
plt.plot(np.arange(0,1024), Net[1,:])


