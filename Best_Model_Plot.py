# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:23:59 2021

@author: gabr8
"""

import pandas
import numpy as np

#Models
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt




#Best Model
Best_model = MLPRegressor(random_state=0, max_iter=10000,
                                      hidden_layer_sizes = Best_params['MLP_hl'], 
                                      learning_rate_init = Best_params['MLP_learning_rate'],
                                      momentum = Best_params['MLP_momentum'],
                                      activation = Best_Params['MLP_activation'],
                                      solver = Best_params['MLP_solver']
                                      )
         
Best_model.fit(X, y)
Best_predict = Best_model.predict(X[0,:])

#unscale the data 
Tproc_predict = inverse_transform(Best_predic)
X_unscale = inverse_transform(X)
y_unscale = inverse_transform(y)

#PLT plotting (MIGHT CHANGE VIZUALISATION LATER)
plt.plot(np.arange(0,1024,1), Best_predict, 'g')
plt.plot(np.arange(0,1024,1), X_unscale[0,:], 'r')
plt.plot(np.arange(0, 1024,1), y, 'b')





#K-fold Cross Validation (For precaution only)
# Errors = -1*cross_val_score(Best_model, X, y, n_jobs = -1 , cv = 5, scoring = 'neg_mean_absolute_error')
# accuracy = Errors.mean()

