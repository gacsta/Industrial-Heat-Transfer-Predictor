# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:30:34 2022

@author: gabr8
"""



import pandas as pd
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np

df_sinal_treino = pd.DataFrame([])
df_sinal_teste = pd.DataFrame([])

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
    df_temp = pd.read_csv(arquivo, sep ='\s+', header = None, usecols=[2]).T
    df_sinal_treino = pd.concat([df_sinal_treino, df_temp], axis=0)
    
    
for arquivo in arquivos_teste:
    df_temp = pd.read_csv(arquivo, sep ='\s+', header = None,usecols=[2]).T
    df_sinal_teste = pd.concat([df_sinal_teste, df_temp], axis=0)
    
    
df_sinal_treino.to_csv('df_sinal_seno_treino.csv')
df_sinal_teste.to_csv('df__sinal_seno_teste.csv')
df_arquivos_teste.to_csv('arquivos_seno_teste.csv')
