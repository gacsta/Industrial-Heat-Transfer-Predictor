# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:22:14 2022

@author: gabr8
"""


import pandas as pd
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np
import re 

'''
--------------------------------> REGRESSION APPROACH DATA TREATMENT <----------------------------------------
'''

folders = [ r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Seno\seno_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Quadrada\quad_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Serrilhado\ser_*',
           r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22-20220927T204627Z-001\Sinais 10-08-22\Triangular\tri_*']


outputs = ['\Seno_reg','\Quad_reg','\Ser_reg', '\Tri_reg']

output_path = r'C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\100_'

tipos = ['seno_(.*)\.','quad_(.*)\.','ser_(.*)\.','tri_(.*)\.']

random.seed(10)
test_size = 0.25
size = 1

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
                
        df_h_treino = pd.DataFrame([],columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf', 'Tau', 'gama', 'Dt',  'h'])
        df_h_teste = pd.DataFrame([],columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf', 'Tau', 'gama', 'Dt', 'h'])

        
        for arquivo in arquivos_treino:
            df_temp = pd.read_csv(arquivo, sep ='\s+', header = None)
            df_temp.columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf', 'Tau', 'gama', 'Dt', 'h']
            
            df_h_treino = pd.concat([df_h_treino, df_temp], axis=0)
            
            
        for arquivo in arquivos_teste:
            df_temp = pd.read_csv(arquivo, sep ='\s+', header = None)
            df_temp.columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf', 'Tau', 'gama', 'Dt', 'h']
            
            df_h_teste = pd.concat([df_h_teste, df_temp], axis=0)
            
            
        df_h_treino.to_csv(output_path + output + '_treino' + '.csv')
        df_h_teste.to_csv(output_path + output + '_teste' + '.csv')
        df_arquivos_teste.to_csv(output_path + output + '_arquivos_teste' + '.csv')
