# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:51:45 2021

@author: gabr8
"""

#DATAFRAME FROM MULTIPLE FILES

import pandas as pd
from glob import glob

semruido = sorted(glob('C:/Users/gabr8/Downloads/ML stuff/sem ruido-20210105T182600Z-001/sem ruido/Sinal_*.txt'))

#Merging files into one single dataframe

Tind = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col=None, usecols = ['Tind']).assign(filename = file) for file in semruido), ignore_index=False, axis = 1)
Tind = Tind.drop(columns = ['filename'], axis = 0)
Tind = Tind.T

Tproc = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col=None, usecols = ['Tproc']).assign(filename = file) for file in semruido), ignore_index=False, axis = 1)
Tproc = Tproc.drop(columns = ['filename'], axis = 0)
Tproc = Tproc.T
