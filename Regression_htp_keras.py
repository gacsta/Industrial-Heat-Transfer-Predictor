# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:06:42 2022

@author: gabr8
"""

# In[ ]:


#The usual collection of indispensables 
import numpy as np
import matplotlib.pyplot as plt
import datetime
# import scipy.fftpack
import pandas as pd

# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.model_selection import KFold
# from tensorflow.keras.layers import BatchNormalization

# LIVE VISUALIZATION
from livelossplot import PlotLossesKeras

from sklearn.preprocessing import StandardScaler


# In[ ]:


def dnn_htp_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.sigmoid,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.RootMeanSquaredError()]) 
  model.summary()
  return model


##########################################################################
##########################################################################
##########################################################################

# --> Try limiting training to low - medium H                ( )
# --> Test without duplicates                                (X)
# --> Play with deeper networks and  higher batches          ( )
# --> Play with different types of data scaling              ( )
# --> Verify behavior after considerable ammount of epochs   ( )

##########################################################################
##########################################################################
##########################################################################



# In[ ]:


df_sinal = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Seno_reg_treino.csv')

# df_quad = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Quad_reg_treino.csv')

# df_tri = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Ser_reg_treino.csv')

# df_ser = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Tri_reg_treino.csv')

# df_sinal = pd.concat([df_seno, df_quad, df_tri, df_ser])
training = df_sinal

training = training.iloc[:, 1:]


# define min max scaler
scaler = StandardScaler()


# In[ ]:

#tind + h
X = training[['Tind','Tind0', 'Tinf', 'gama', 'h']] 
X = pd.DataFrame(scaler.fit_transform(X))


y = training['Tproc'] 



# In[ ]:


EPOCHS = 1000
# model = dnn_htp_model()
strt_time = datetime.datetime.now()
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
from tensorflow.keras.models import load_model
model = load_model(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\temp')
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
history = model.fit(X, y, 
                    epochs=EPOCHS,  
                    validation_split=0.25, verbose=1,
                    # callbacks=[PlotLossesKeras(), callback], 
                    
                    batch_size = 32
                    )
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
plt.show()


# In[ ]:

model.save(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\temp')


df2 = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Tri_reg_teste.csv')

# df_quad = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Quad_reg_teste.csv')

# df_tri = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Ser_reg_teste.csv')

# df_ser = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Tri_reg_teste.csv')

# df2 = pd.concat([df_seno, df_quad, df_tri, df_ser])

df2 = df2.iloc[:, 1:]
# df2 = df2.drop_duplicates(subset = ['Tind','Tind0', 'Tinf', 'gama', 'h', 'Tproc'])
# df2.reset_index(drop = True, inplace = True)

# In[ ]:

#tind + h
X_test = df2[['Tind','Tind0', 'Tinf', 'gama', 'h']] 
X_test = scaler.transform(X_test)
y_test = df2['Tproc']

# tind_plot = df2.iloc[:,130]

# In[ ]:
from sklearn.metrics import r2_score,mean_squared_error

resultados = pd.DataFrame([], columns = ['H', 'RMSE', 'R2', 'sinais_rep'])

y_pred = model.predict(X_test).flatten()

keras_dnn_err = y_test - y_pred

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = True)

r2_reconstruida = r2_score(y_test, df2['Trec'])
rmse_reconstruido = mean_squared_error(y_test, df2['Trec'], squared = True)

for num in np.arange(0, int(len(X_test)/1024)):
    
    keras_dnn_err_h = y_test[(num*1024):((num*1024)+1024)] - y_pred[(num*1024):((num*1024)+1024)]   
    r2_h = r2_score(y_test[(num*1024):((num*1024)+1024)], y_pred[(num*1024):((num*1024)+1024)])
    rmse_h = mean_squared_error(y_test[(num*1024):((num*1024)+1024)], y_pred[(num*1024):((num*1024)+1024)], squared = True)
    sinais_rep = len(df2[df2['h'] == df2['h'][num*1024]])/1024
    h = df2['h'][num*1024]
    
    rmse_rec = mean_squared_error(y_test[(num*1024):((num*1024)+1024)], df2['Trec'][(num*1024):((num*1024)+1024)], squared = True)
    r2_rec = r2_score(y_test[(num*1024):((num*1024)+1024)], df2['Trec'][(num*1024):((num*1024)+1024)])
    
    resultados = pd.concat([resultados, pd.DataFrame([[h, rmse_h, r2_h, sinais_rep, rmse_rec, r2_rec]] , columns = ['H', 'RMSE', 'R2', 'sinais_rep', 'RMSE_rec', 'R2_rec'])], axis = 0)
    
    fig = plt.figure(figsize = (20,15))
    
    
    plt.plot(np.arange(0,1024), df2['Trec'][(num*1024):((num*1024)+1024)], 'y', label = 'Trec')
    plt.plot(np.arange(0,1024),y_test[(num*1024):((num*1024)+1024)], 'b', label = 'Tproc')
    # plt.plot(np.arange(0,1024),y_pred[(num*1024):((num*1024)+1024)], 'g', label = 'Trede')
    

    
    plt.plot(np.arange(0,1025), df2['Tind'].loc[(num*1024):((num*1024)+1024)], 'r', label = 'Tind')
    m, b = np.polyfit(np.arange(0,65),  df2['Tind'].loc[(3*1024):((3*1024)+64)], 1)
    #add linear regression line to scatterplot 
    plt.plot(np.arange(0,65), (m*(np.arange(0,65))+b))
    plt.legend(loc="upper left")
    # plt.savefig(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\Resultados\Ser\H_{}_rmse_{}_h_{}.png'.format(h, rmse_h,r2_h))
    plt.show()
    
# resultados.to_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\Resultados\Ser\modelo\resultados_ser_reg.csv')
# In[ ]:

model.save(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\temp')



# In[ ]:





# In[ ]:





# In[ ]:




