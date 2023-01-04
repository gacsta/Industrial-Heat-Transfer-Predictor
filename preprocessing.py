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

df_h_treino = pd.DataFrame([],columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf','Tau', 'gama', 'Dt', 'h'])
df_h_teste = pd.DataFrame([],columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf','Tau', 'gama', 'Dt', 'h'])

mypath = r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\seno\seno_*'
arquivos = sorted(glob(mypath))

test_size = 0.33
indices = sorted(np.arange(1,len(arquivos)))
indices_teste = random.sample(indices, round(len(arquivos)*test_size))

arquivos_teste = [arquivos[i] for i in indices_teste]
arquivos_treino = [arquivos[i] for i in indices if i not in indices_teste]

import re 
df_arquivos_teste = pd.DataFrame(arquivos_teste,columns=['path'])
df_arquivos_teste['num'] = [int(re.findall('seno_(.*)\.', path)[0]) for path in df_arquivos_teste['path']]
df_arquivos_teste.sort_values(by=['num'], inplace = True)

for arquivo in arquivos_treino:
    df_temp = pd.read_csv(arquivo, sep ='\s+', header = None)
    df_temp.columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf','Tau', 'gama', 'Dt', 'h']
    
    df_h_treino = pd.concat([df_h_treino, df_temp], axis=0)
    
    
for arquivo in arquivos_teste:
    df_temp = pd.read_csv(arquivo, sep ='\s+', header = None)
    df_temp.columns = ['Time','Tproc', 'Tind', 'Trec', 'Tind0', 'Tinf','Tau', 'gama', 'Dt', 'h']
    
    df_h_teste = pd.concat([df_h_teste, df_temp], axis=0)
        

df_h_treino_ndp = df_h_treino.drop_duplicates(subset = ['Tproc'])
    
df_h_treino.to_csv('df_h_seno_treino.csv')
df_h_teste.to_csv('df__seno_h_teste.csv')
df_arquivos_teste.to_csv('arquivos_seno_teste.csv')
