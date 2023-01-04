# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:50:06 2022

@author: gabr8
"""

import pandas as pd
from glob import glob
import random
import numpy as np

folders = [r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22\Seno\seno_*',
           r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22\Quadrada\quad_*',
           r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22\Serrilhado\ser_*',
           r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Sinais 10-08-22\Triangular\tri_*']

outputs = [r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\Concatenado\Seno2808.csv',
          r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\Concatenado\Quad2808.csv',
          r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\Concatenado\Ser2808.csv',
          r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\Concatenado\Tri2808.csv']

for folder,output  in zip(folders, outputs):
        #####################################################################################################################################
        
        mypath = folder
        arquivos = sorted(glob(mypath))
        
        #####################################################################################################################################
        # arquivos_treino = arquivos
        #####################################################################################################################################
        
        test_size = 0.2
        indices = sorted(np.arange(0,len(arquivos)))
        indices_teste = random.sample(indices, round(len(arquivos)*test_size))
        
        arquivos_teste = [arquivos[i] for i in indices_teste]
        arquivos_treino = [arquivos[i] for i in indices if i not in indices_teste]
        
        #####################################################################################################################################
        #ORDENANDO PATHS POR H DE SEUS RESPECTIVOS SINAIS
        import re 
        
        df_arquivos_teste = pd.DataFrame(arquivos_teste,columns=['path'])
        df_arquivos_teste['num'] = [int(re.findall('seno_(.*)\.', path)[0]) for path in df_arquivos_teste['path']]
        df_arquivos_teste.sort_values(by=['num'], inplace = True)
        
        #####################################################################################################################################
        
        df_arquivos_treino = pd.DataFrame(arquivos_treino,columns=['path'])
        df_arquivos_treino['num'] = [int(re.findall('seno_(.*)\.', path)[0]) for path in df_arquivos_treino['path']]
        df_arquivos_treino.sort_values(by=['num'], inplace = True)
        
        #####################################################################################################################################
        
        arquivos_teste = df_arquivos_teste['path']
        arquivos_treino = df_arquivos_treino['path']
        
        arquivos_treino.reset_index(drop = True, inplace = True)
        arquivos_teste.reset_index(drop = True, inplace = True)
        
        
        window = 129
        
        
        df_sinal = pd.DataFrame([])
        arquivos_treino = arquivos_treino[:200]
        
        
        for arquivo in arquivos_treino:
            
                sinal = pd.read_csv(arquivo ,sep ='\s+', header = None, usecols = [1,2,9])
                
                for w in range(0, len(sinal)+1-window):
                    
                    transpose = sinal[2].iloc[0+w:128+w]
                    
                    #ADD TPROC
                    transpose = pd.concat([transpose, pd.Series(sinal[1].iloc[128+w])], axis = 0, ignore_index = True)
                    
                    #ADD H
                    transpose = pd.concat([transpose, pd.Series(sinal[9].iloc[128+w])], axis = 0, ignore_index = True) 
                    
                    #ADD TIND FOR PLOTTING REASONS
                    transpose = pd.concat([transpose, pd.Series(sinal[2].iloc[128+w])], axis = 0, ignore_index = True)       
                    
        
                    
                    transpose = pd.DataFrame([transpose])
                    
                    df_sinal = pd.concat([df_sinal,transpose], axis = 0)
                    
                    
        df_sinal.reset_index(drop=True, inplace = True)           
        df_sinal.to_csv(r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Train\df_seno_treino_128_h.csv')
            
        
        
        arquivos_teste = arquivos_teste[:50]
        
        df_sinal = pd.DataFrame([])
        
        num = 0
        for arquivo in arquivos_teste:
                
                df_sinal_teste = pd.DataFrame([])
                
                sinal = pd.read_csv(arquivo ,sep ='\s+', header = None, usecols = [1,2,9])
            
                for w in range(0, len(sinal)+1-window):
                    
                    transpose = sinal[2].iloc[0+w:128+w]
                    
                    #ADD TPROC
                    transpose = pd.concat([transpose, pd.Series(sinal[1].iloc[128+w])], axis = 0, ignore_index = True)
                    
                    #ADD H
                    transpose = pd.concat([transpose, pd.Series(sinal[9].iloc[128+w])], axis = 0, ignore_index = True) 
                    
                    #ADD TIND FOR PLOTTING REASONS
                    transpose = pd.concat([transpose, pd.Series(sinal[2].iloc[128+w])], axis = 0, ignore_index = True)       
                    
                    transpose = pd.DataFrame([transpose])
                    
                    
                    df_sinal = pd.concat([df_sinal,transpose], axis = 0)
                    df_sinal_teste = pd.concat([df_sinal_teste,transpose], axis = 0)
                    
                df_sinal_teste.to_csv(r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\df_seno_sinal_teste_{num}.csv'.format(num = num)) 
                num+=1
                    
                   
                    
                
        df_sinal.to_csv(r'C:\Users\gabr8\OneDrive\Documentos\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\signal_approach\Test\Concatenado\df_seno_teste_128_h.csv')
            
        
            
