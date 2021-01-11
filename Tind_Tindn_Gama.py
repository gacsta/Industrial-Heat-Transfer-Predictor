# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:30:30 2021

@author: gabr8
"""


#DATAFRAME FROM MULTIPLE LVM FILES 

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

semruido = sorted(glob('C:/Users/gabr8/Downloads/ML stuff/drive-download-20210108T200102Z-002/Sinais 05-11-2019/Quadrada/quad_*.lvm'))

#Merging files into one single dataframe
#FIGURE OUT A WAY TO TRANSPOSE THE FEATURE COLUMNS 3 COLUMNS --->ONE ROW
Inputs = pd.concat((pd.read_csv('C:/Users/gabr8/Downloads/ML stuff/drive-download-20210108T200102Z-002/Sinais 05-11-2019/Quadrada/quad_3.lvm', encoding ='latin1', sep="\s+", index_col= 0, header = None, names = ['Tproc', 'Tind', 'Trec', 'Tind-1', 'Tinf', 'Tau', 'Gama', 'Deltat']).assign(filename = file) for file in semruido), ignore_index=False, axis = 1)
Inputs = Inputs.drop(columns = ['filename', 'Tproc', 'Tinf', 'Tau', 'Deltat', 'Trec'], axis = 0)


# Tproc = pd.concat((pd.read_csv(file, encoding ='latin1', sep="\s+", index_col=None, usecols = ['Tproc']).assign(filename = file) for file in semruido), ignore_index=False, axis = 1)
# Tproc = Tproc.drop(columns = ['filename'], axis = 0)
# Tproc = Tproc.T


# test = pd.read_csv('C:/Users/gabr8/Downloads/ML stuff/drive-download-20210108T200102Z-002/Sinais 05-11-2019/Quadrada/quad_3.lvm', encoding ='latin1', sep="\s+", index_col= 0, header = None, names = ['Tproc', 'Tind', 'Trec', 'Tind-1', 'Tinf', 'Tau', 'Gama', 'Deltat'] )    
# features = ['Tind', 'Tind-1', 'Gama']
# testX = test[features]
# X = pd.concat([testX['Tind'], testX['Tind-1'], testX['Gama']], ignore_index= True, join = 'inner')

# X = X.transpose()

# test['Tproc'].plot(color = 'b')
# test['Tind'].plot(color = 'g')

