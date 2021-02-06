# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:05:23 2021

@author: gabr8
"""


import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#Pipelines
from sklearn.pipeline import Pipeline

#GET_DATA_FROM
from Tind_Tindn_Gama import *

# #Splitting the data (Without K-fold cross val)
# X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 0)


# ----> NEEDS ADAPTATION TO PIPELINE STRUCTURE <----
#Prmimitive parameter testing
def model_tester(X, y, hl):      
    
    #Generating pipeline
    rede_pipeline = Pipeline(steps = [('mlp', MLPRegressor(random_state=0, max_iter=10000, hidden_layer_sizes = hl))])
    
    #K-fold Cross Validation
    Errors = -1*cross_val_score(rede_pipeline, X, y, cv = 5, scoring = 'neg_mean_absolute_error')
    Mean_Error = Errors.mean()
    Net = cross_val_predict(rede_pipeline, X,y, cv = 5)
    
    return Net, hl, Mean_Error

#hidden layer sizes
#500 --> 2000
hl = np.arange(1000,1200, 50)
Model = []

for i in hl:

            Model.append(model_tester(X, y, (i,)))
                    
            
   






