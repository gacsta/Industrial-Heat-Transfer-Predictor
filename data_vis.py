# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:15:53 2022

@author: gabr8
"""



import numpy as np
import matplotlib.pyplot as plt
import datetime
# import scipy.fftpack
import pandas as pd

mult_quad = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Quad\modelo\resultados_quad_regmult.csv")
mult_seno = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Seno\modelo\resultados_seno_regmult.csv")
# mult_ser = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Ser\modelo\resultados_ser_regmult.csv")
mult_tri = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Tri\modelo\resultados_tri_regmult.csv")

 
series_quad = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\Quad\modelo\resultados_quad_time_series.csv")
series_seno = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\Seno\modelo\resultados_seno_time_series.csv")
series_ser = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\Ser\modelo\resultados_ser_time_series.csv")
# series_tri = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\TimeSeries Approach\100\RESULTADOS\Tri\modelo\resultados_ser_time_series.csv")

# reg_quad = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Quad\modelo\resultados_quad_regmult.csv")
reg_seno = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\Resultados\Seno\modelo\resultados_seno_reg.csv")
reg_ser = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Approach\Resultados\Ser\modelo\resultados_ser_reg.csv")
# reg_tri = pd.read_csv(r"C:\Users\gabr8\Documents\GitHub\Industrial-Heat-Transfer-Predictor\SinaisDEZ_hvariante\Sinais 10-08-22-20220815T211356Z-001\Preprocessed\Regression Multivariate\Tri\modelo\resultados_quad_regmult.csv")

 

#########################################################################################################################
# FULL H SPECTRUM RESULTS
#########################################################################################################################

# MULTIVARIATE POST-PROCESSING

mult_med_RMSE = np.mean([np.mean(mult_seno["RMSE"]),np.mean(mult_seno["RMSE"]),np.mean(mult_tri["RMSE"])])
mult_med_RMSE_rec = np.mean([np.mean(mult_seno["RMSE_rec"]),np.mean(mult_seno["RMSE_rec"]),np.mean(mult_tri["RMSE_rec"])])


mult_med_R2 = np.mean([np.mean(mult_seno["R2"]),np.mean(mult_seno["R2"]),np.mean(mult_tri["R2"])])
mult_med_R2_rec = np.mean([np.mean(mult_seno["R2_rec"]),np.mean(mult_seno["R2_rec"]),np.mean(mult_tri["R2_rec"])])

# TIME-SERIES
series_med_RMSE = np.mean([np.mean(series_seno["RMSE"]),np.mean(series_ser["RMSE"])])
series_med_RMSE_rec = np.mean([np.mean(series_seno["RMSE_rec"]),np.mean(series_ser["RMSE_rec"])])


series_med_R2 = np.mean([np.mean(series_seno["R2"]),np.mean(series_ser["R2"])])
series_med_R2_rec = np.mean([np.mean(series_seno["R2_rec"]),np.mean(series_ser["R2_rec"])])

#REGRESSION
reg_med_RMSE = np.mean([np.mean(reg_ser["RMSE"]),np.mean(reg_ser["RMSE"])])
reg_med_RMSE_rec = np.mean([np.mean(reg_ser["RMSE_rec"]),np.mean(reg_ser["RMSE_rec"])])


reg_med_R2 = np.mean([np.mean(reg_ser["R2"]),np.mean(reg_ser["R2"])])
reg_med_R2_rec = np.mean([np.mean(reg_ser["R2_rec"]),np.mean(reg_ser["R2_rec"])])
 
#########################################################################################################################
# SUB 5000 H RESULTS
#########################################################################################################################


# MULTIVARIATE POST-PROCESSING

mult_med_RMSE_sub5000  = np.mean([np.mean(mult_seno["RMSE"][mult_seno["H"] < 5000 ]),np.mean(mult_seno["RMSE"][mult_seno["H"] < 5000 ]),np.mean(mult_tri["RMSE"][mult_tri["H"] < 5000 ])])
mult_med_RMSE_rec_sub5000  = np.mean([np.mean(mult_seno["RMSE_rec"][mult_seno["H"] < 5000 ]),np.mean(mult_seno["RMSE_rec"][mult_seno["H"] < 5000 ]),np.mean(mult_tri["RMSE_rec"][mult_tri["H"] < 5000 ])])


mult_med_R2_sub5000  = np.mean([np.mean(mult_seno["R2"][mult_seno["H"] < 5000 ]),np.mean(mult_seno["R2"][mult_seno["H"] < 5000 ]),np.mean(mult_tri["R2"][mult_tri["H"] < 5000 ])])
mult_med_R2_rec_sub5000  = np.mean([np.mean(mult_seno["R2_rec"][mult_seno["H"] < 5000 ]),np.mean(mult_seno["R2_rec"][mult_seno["H"] < 5000 ]),np.mean(mult_tri["R2_rec"][mult_tri["H"] < 5000 ])])

# TIME-SERIES
series_med_RMSE_sub5000  = np.mean([np.mean(series_seno["RMSE"][series_seno["H"] < 5000 ]),np.mean(series_ser["RMSE"][series_ser["H"] < 5000 ])])
series_med_RMSE_rec_sub5000  = np.mean([np.mean(series_seno["RMSE_rec"][series_seno["H"] < 5000 ]),np.mean(series_ser["RMSE_rec"][series_ser["H"] < 5000 ])])


series_med_R2_sub5000  = np.mean([np.mean(series_seno["R2"][series_seno["H"] < 5000 ]),np.mean(series_ser["R2"][series_ser["H"] < 5000 ])])
series_med_R2_rec_sub5000  = np.mean([np.mean(series_seno["R2_rec"][series_seno["H"] < 5000 ]),np.mean(series_ser["R2_rec"][series_ser["H"] < 5000 ])])


#Regression
reg_med_RMSE_sub5000  = np.mean([np.mean(reg_ser["RMSE"][reg_ser["H"] < 5000 ]),np.mean(reg_ser["RMSE"][reg_ser["H"] < 5000 ])])
reg_med_RMSE_rec_sub5000  = np.mean([np.mean(reg_ser["RMSE_rec"][reg_ser["H"] < 5000 ]),np.mean(reg_ser["RMSE_rec"][reg_ser["H"] < 5000 ])])


reg_med_R2_sub5000  = np.mean([np.mean(reg_ser["R2"][reg_ser["H"] < 5000 ]),np.mean(reg_ser["R2"][reg_ser["H"] < 5000 ])])
reg_med_R2_rec_sub5000  = np.mean([np.mean(reg_ser["R2_rec"][reg_ser["H"] < 5000 ]),np.mean(reg_ser["R2_rec"][reg_ser["H"] < 5000 ])])


