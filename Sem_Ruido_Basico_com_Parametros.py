# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:46:18 2021

@author: gabr8
"""

import numpy as np
import matplotlib.pyplot as plt

#Geração do sinal
tempo=10/262143
amp=100
Tindn=100+273
t = np.arange(0, 10, tempo)
M = 4.7*10**(-6)
C = 3.8*(10**2)
h = 5.5*(10**2)
A = 3.14*10**(-6)
tau=(M*C)/(h*A)

Tempo = np.full(len(t), tempo)
Tinf = np.full(len(t), Tindn)
Tau = np.full(len(t), tau)

epsilon=1
sigma=5.670*10**(-8)
Tproc= np.sin(t)+amp+273

# gama = np.full(len(t), 0)

# Tind = np.full(len(t), 0)


gama = np.full(len(t),((4*epsilon*sigma)/h)*((Tindn-Tindn)/2)**3)
Tind = np.full(len(t),(1/((tau/tempo)+1+gama[0])*((Tproc[0])+(gama[0]*Tindn)+((tau/tempo)*Tindn))))

for i in range(1,len(t)):
   
    gama[i]=((4*epsilon*sigma)/h)*((Tindn-Tind[i-1])/2)**3
    Tind[i]=(1/((tau/tempo)+1+gama[i])*((Tproc[i])+(gama[i]*Tindn)+((tau/tempo)*Tind[i-1])))
    
    
# for i in range(0, len(t)):
#     Tind[i] = Tind[i] + np.random.uniform(-0.000005,0.000005)


Trec = np.full(len(t), Tind[1]+(tau/tempo)*(Tind[1]-Tindn)-gama[1]*(Tindn-Tindn))

for i in range(1, len(t)):
    Trec[i]=Tind[i]+(tau/tempo)*(Tind[i]-Tind[i-1])-gama[i]*(Tindn-Tind[i-1]);

plt.plot(t,Tproc,t,Tind,t,Trec)
