# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:51:22 2021

@author: gabr8
"""

# import pandas as pd

#Models
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split


#Hyperparameter Optmization
import optuna

#GET_DATA_FROM
import Reader_Sinais_mCheA as rs

#Splitting the data (Without K-fold cross val)
X_train, X_val, y_train, y_val = train_test_split(rs.X, rs.y, train_size = 0.7, test_size = 0.3, random_state = 0)



###ADAM SOLVER


# 1. Define an objective function to be maximized.
def objective(trial):


        MLP_hl = trial.suggest_int('MLP_hl', 10, 500)
        MLP_activation = trial.suggest_categorical('MLP_activation', ['identity', 'logistic', 'tanh', 'relu'])
        MLP_learning_rate_init = trial.suggest_float('MLP_learning_rate_init', 0.00001, 0.1, log = True)
        MLP_beta_1 = trial.suggest_float('MLP_beta_1', 0.01, 1, log = True)
        MLP_beta_2 = trial.suggest_float('MLP_beta_1', 0.01, 1, log = True)
        MLP_epsilon = trial.suggest_float('MLP_epsilon', 1e-7, 10e-3, log=True)
        MLP_alpha = trial.suggest_float('MLP_alpha', 1e-7, 1e-5, log = True)
        
        classifier_obj = MLPRegressor(random_state=0, max_iter=10000,
                                      hidden_layer_sizes = (MLP_hl,),
                                      learning_rate_init = MLP_learning_rate_init,
                                      activation = MLP_activation,
                                      beta_1 = MLP_beta_1,
                                      beta_2 = MLP_beta_2,
                                      epsilon = MLP_epsilon,
                                      alpha = MLP_alpha,
                                      batch_size = 1,
                                      solver = 'adam'
                                      ).fit(X_train, y_train)
        
        accuracy = classifier_obj.score(X_val, y_val)
         
        # #K-fold Cross Validation
        # Errors = -1*cross_val_score(classifier_obj, X, y, n_jobs = -1 , cv = 5, scoring = 'neg_mean_absolute_error')
        # accuracy = Errors.mean()
        
        
        

        return accuracy

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=None)
Best = study.best_trial
Best_params = study.best_params


            
   






