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
# from livelossplot import PlotLossesKeras


# In[ ]:
    
#  if not os.path.exists('tf_keras_cifar10.h5'):
#     model = get_model() #this method constructs the model and compiles it 
# else:
#     model = load_model('tf_keras_cifar10.h5') #load the model from file
#     print('lr is ', K.get_session().run(model.optimizer.lr))
#     initial_epoch=10
#     epochs=13


def dnn_htp_model():
  model = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.RootMeanSquaredError()]) 
  model.summary()
  return model


# In[ ]:
    
df_sinal = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Seno_reg_treino.csv')

# df_sinal = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\Quad2808_treino.csv')

training = df_sinal

training = training.iloc[:, 1:]

# test = training[training['H'] < 2000]

# define min max scaler
# scaler = MinMaxScaler()


# In[ ]:

#tind + h
# X = pd.concat([training.iloc[:,0:128], training.iloc[:,129]], axis = 1)
# X = pd.DataFrame(scaler.fit_transform(X))
X = training[['Tind','Tind0','Tinf','Tau','gama','h']]
y = training['Tproc']
#apenas tind
#X = df_sinal.iloc[:,0:128]

# y = training.iloc[:,128]



# In[ ]:


EPOCHS = 100000
##################################################
##################################################
model = dnn_htp_model()
##################################################
##################################################
from tensorflow.keras.models import load_model
# model = load_model(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\temp')
model.summary()
epoch = 50000
# initial_epoch = 1397
##################################################
##################################################
strt_time = datetime.datetime.now()
history = model.fit(X, y, 
                    epochs=EPOCHS,
                    validation_split=0.3, verbose= 1,
                    # initial_epoch = initial_epoch,
                    # callbacks=[]               ;
                    batch_size = 32
                    # callbacks=[PlotLossesKeras()]
                    )
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
plt.show()


# In[ ]:


# num = 40
# df2 = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\Quad2808_teste.csv')


df2 = pd.read_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_\Seno_reg_teste.csv')
df2 = df2.iloc[:, 1:]
# df2 = df2[df2['H'] < 2000] 


# In[ ]:

#tind + h
X_test = pd.concat([df2.iloc[:,0:128], df2.iloc[:,129]], axis = 1)
Trec = df2['Trec']
y_test = df2.iloc[:,128]

# tind_plot = df2.iloc[:,130]

# In[ ]:
from sklearn.metrics import r2_score,mean_squared_error

resultados = pd.DataFrame([], columns = ['H', 'RMSE', 'R2', 'sinais_rep'])

y_pred = model.predict(X_test).flatten()

keras_dnn_err = y_test - y_pred

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = True)

r2_reconstruida = r2_score(y_test, Trec)
rmse_reconstruido = mean_squared_error(y_test, Trec, squared = True)

for num in np.arange(0, int(len(X_test)/896)):
    
    keras_dnn_err_h = y_test[(num*896):((num*896)+896)] - y_pred[(num*896):((num*896)+896)]   
    r2_h = r2_score(y_test[(num*896):((num*896)+896)], y_pred[(num*896):((num*896)+896)])
    rmse_h = mean_squared_error(y_test[(num*896):((num*896)+896)], y_pred[(num*896):((num*896)+896)], squared = True)
    sinais_rep = len(X_test[X_test['H'] == X_test['H'][num*896]])/896
    h = X_test['H'].loc[num*896]
    
    rmse_rec = mean_squared_error(y_test[(num*896):((num*896)+896)], Trec[(num*896):((num*896)+896)], squared = True)
    r2_rec = r2_score(y_test[(num*896):((num*896)+896)], Trec[(num*896):((num*896)+896)])
    
    resultados = pd.concat([resultados, pd.DataFrame([[h, rmse_h, r2_h, sinais_rep, rmse_rec, r2_rec]] , columns = ['H', 'RMSE', 'R2', 'sinais_rep', 'RMSE_rec', 'R2_rec'])], axis = 0)
    
    fig = plt.figure()
    
    plt.plot(np.arange(0,896), Trec[(num*896):((num*896)+896)], 'y')
    plt.plot(np.arange(0,896),y_test[(num*896):((num*896)+896)], 'b')
    plt.plot(np.arange(0,896),y_pred[(num*896):((num*896)+896)], 'g')
    plt.plot(np.arange(0,896),X_test['2'].loc[(num*896):((num*896)+895)], 'r')
    
    # plt.savefig(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\H_{}_rmse_{}_h_{}.png'.format(h, rmse_h,r2_h))
    plt.show()
    
# resultados.to_csv(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\Quad\resultados_quad_time_series.csv')
# In[ ]:

model.save(r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\temp')



# In[ ]:





# In[ ]:





# In[ ]:




