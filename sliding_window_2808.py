# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:50:07 2022

@author: gabr8
"""

import pandas as pd
from glob import glob
import random
import numpy as np

import matplotlib.pyplot as plt

import itertools
import re 


folders = [ r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Seno\seno_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Quadrada\quad_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Serrilhado\ser_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Triangular\tri_*']

outputs = ['\Seno2808','\Quad2808','\Ser2808', '\Tri2808']

output_path = r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\256'

tipos = ['seno_(.*)\.','quad_(.*)\.','ser_(.*)\.','tri_(.*)\.']

size = 1;
window = 256
test_size = 0.33

for folder,output,tipo  in zip(folders, outputs, tipos):
        
        #####################################################################################################################################
        
        mypath = folder
        arquivos = sorted(glob(mypath))
        arquivos = random.sample(arquivos, int(len(arquivos)*size))
        
        indices = sorted(np.arange(1,len(arquivos)))
        indices_teste = random.sample(indices, round(len(arquivos)*test_size))

        arquivos_teste = [arquivos[i] for i in indices_teste]
        arquivos_treino = [arquivos[i] for i in indices if i not in indices_teste]
        
        #####################################################################################################################################
        #ORDENANDO PATHS POR H DE SEUS RESPECTIVOS SINAIS

        
        #TREINO
        df_arquivos_treino = pd.DataFrame(arquivos_treino,columns=['path'])
        df_arquivos_treino['num'] = [int(re.findall(tipo, path)[0]) for path in df_arquivos_treino['path']]
        df_arquivos_treino.sort_values(by=['num'], inplace = True)
        
        arquivos_treino = df_arquivos_treino['path']
        arquivos_treino.reset_index(drop = True, inplace = True)
        #TESTE
        df_arquivos_teste = pd.DataFrame(arquivos_teste,columns=['path'])
        df_arquivos_teste['num'] = [int(re.findall(tipo, path)[0]) for path in df_arquivos_teste['path']]
        df_arquivos_teste.sort_values(by=['num'], inplace = True)
        
        arquivos_teste = df_arquivos_teste['path']
        arquivos_teste.reset_index(drop = True, inplace = True)
        
        #####################################################################################################################################
        #####################################################################################################################################
        
        data_treino = pd.DataFrame([])
        data_teste = pd.DataFrame([])
        
        
        for arquivo_treino, arquivo_teste in itertools.zip_longest(arquivos_treino, arquivos_teste):
            # treino
            #####################################################################################################################################
            X_treino = pd.DataFrame([])
            sinal_treino = pd.read_csv(arquivo_treino ,sep ='\s+', header = None)
            
            X_treino = pd.concat([sinal_treino[2].shift(-1*(i - 1)) for i in range(1, window+1)], axis = 1)
            
            X_treino['Tproc'] = sinal_treino[1].shift(-window)
            X_treino['H'] = sinal_treino[9]
            X_treino['Trec'] = sinal_treino[3].shift(-window)
            X_treino.dropna(axis = 0, inplace = True)  
            data_treino = pd.concat([data_treino, X_treino], axis = 0)
        
            #####################################################################################################################################
            # teste
            #####################################################################################################################################
            if arquivo_teste is not None:
                X_teste = pd.DataFrame([])
                sinal_teste = pd.read_csv(arquivo_teste ,sep ='\s+', header = None)
                
                X_teste = pd.concat([sinal_teste[2].shift(-1*(i - 1)) for i in range(1, window+1)], axis = 1)
                
                X_teste['Tproc'] = sinal_teste[1].shift(-window)
                X_teste['H'] = sinal_teste[9]
                X_teste['Trec'] = sinal_teste[3].shift(-window)
                X_teste.dropna(axis = 0, inplace = True)  
                data_teste = pd.concat([data_teste, X_teste], axis = 0)
            #####################################################################################################################################
            
            
            
        data_treino.to_csv(output_path + output + '_treino'+'.csv') 
        data_teste.to_csv(output_path + output + '_teste'+'.csv')
            


