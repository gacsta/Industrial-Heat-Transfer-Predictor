# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:51:22 2021

@author: gabr8
"""

#Models
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict


#Hyperparameter Optmization
import optuna

#GET_DATA_FROM
import Reader_Sinais_mCheA as rs 


###SGD SOLVER


# 1. Define an objective function to be maximized.
def objective(trial):


        MLP_hl = trial.suggest_int('MLP_hl', 5, 500)
        MLP_activation = trial.suggest_categorical('MLP_activation', ['identity', 'logistic', 'tanh', 'relu'])
        MLP_learning_rate_init = trial.suggest_float('MLP_learning_rate_init', 0.00001, 0.1, log = True)
        MLP_learning_rate = trial.suggest_categorical('MLP_learning_rate', ['constant', 'invscaling', 'adaptive'])
        MLP_momentum = trial.suggest_float('MLP_momentum', 0.1, 1, log = True)
        MLP_alpha = trial.suggest_float('MLP_alpha', 1e-8, 1e-3, log = True)
        
        classifier_obj = MLPRegressor(random_state=0, max_iter=1000,
                                      hidden_layer_sizes = (MLP_hl,), 
                                      learning_rate_init = MLP_learning_rate_init,
                                      learning_rate = MLP_learning_rate,
                                      momentum = MLP_momentum,
                                      activation = MLP_activation,
                                      alpha = MLP_alpha,
                                      solver = 'sgd'
                                      ).fit(rs.X_train, rs.y_train)
        
       
        accuracy = classifier_obj.score(rs.X_val, rs.y_val)
        
        # #K-fold Cross Validation
        # Errors = -1*cross_val_score(classifier_obj, X, y, n_jobs = -1 , cv = 5, scoring = 'neg_mean_absolute_error')
        # accuracy = Errors.mean()
        

        return accuracy

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=None)
Best = study.best_trial
Best_params = study.best_params


            
   






