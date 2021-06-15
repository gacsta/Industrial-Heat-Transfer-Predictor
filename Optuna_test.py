# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:51:22 2021

@author: gabr8
"""

#Models
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pickle

#Hyperparameter Optmization
import optuna

#GET_DATA_FROM
with open(r'C:\Users\Unesp\Documents\GitHub\Industrial-Heat-Transfer-Predictor\drive-download-20210603T192100Z-001/X_train', 'rb') as pasta:
    X_train = pickle.load(pasta)
with open(r'C:\Users\Unesp\Documents\GitHub\Industrial-Heat-Transfer-Predictor\drive-download-20210603T192100Z-001/y_train', 'rb') as pasta:
    y_train = pickle.load(pasta)



###SGD SOLVER


def objective(trial):
    
        n_layers = trial.suggest_int('n layers ', 2, 4)
    
        layers = []
        for i in range(n_layers):
            
            layers.append(trial.suggest_int('n units - layer {} '.format(i), 100, 4000))


        #MLP_hl = trial.suggest_int('MLP_hl', 100, 5000)
        MLP_activation = trial.suggest_categorical('MLP_activation', ['logistic', 'tanh'])
        MLP_learning_rate_init = trial.suggest_float('MLP_learning_rate_init', 1e-7, 1e-4, log = True)
        MLP_learning_rate = trial.suggest_categorical('MLP_learning_rate', ['constant', 'invscaling', 'adaptive'])
        MLP_momentum = trial.suggest_float('MLP_momentum', 0.0, 1)
        MLP_alpha = trial.suggest_float('MLP_alpha', 1e-7, 1e-1, log = True)
        MLP_power_t = trial.suggest_float("MLP_power_t", 0.2, 0.8, step=0.1)
        
        classifier_obj = MLPRegressor(random_state=0, max_iter=3000,
                                      hidden_layer_sizes = (tuple(layers)),
                                      learning_rate_init = MLP_learning_rate_init,
                                      learning_rate = MLP_learning_rate,
                                      momentum = MLP_momentum,
                                      activation = MLP_activation,
                                      power_t = MLP_power_t,
                                      alpha = MLP_alpha,
                                      solver = 'sgd'
                                      ).fit(X_train, y_train)
        
        #K-fold Cross Validation
        Errors = -1*cross_val_score(classifier_obj, X_train, y_train, n_jobs = -1 , cv = 5, scoring = 'neg_mean_absolute_error')
        accuracy = Errors.mean()

        if accuracy < 40:
              with open("Lab2_Modelor2r2_{}.pickle".format(trial.number), "wb") as fout:
                  pickle.dump(classifier_obj, fout)
              with open("Lab2_Parametersr2_{}.pickle".format(trial.number), "wb") as fout:
                  pickle.dump(trial.params, fout)
              with open("Lab2_Study.pickle", "wb") as fout:
                  pickle.dump(study, fout)
              
             
        

        return accuracy
# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=None)
Best = study.best_trial
Best_params = study.best_params

                    
                     
                     
            
   






