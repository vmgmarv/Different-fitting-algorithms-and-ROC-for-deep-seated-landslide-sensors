# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:56:00 2019
"""

import pandas as pd
import numpy as np
import filterdata as filt
from scipy.signal import savgol_filter
import mysql.connector as sql
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.cm as cm
from scipy import stats, polyval
sns.set_style("darkgrid")



def get_features(array):
    
    x = []
    y = []
    z = []
    
    m = 144 ##### 3 day data
    t = 145 ##### for extension
    
    for i in range(len(array) - t):
        
        x.append(array[i:i+m])
        y.append(array[i+m])
        z.append(array[i+t] - array[i+m]) ################### Check if the next time data moves
        
    x = np.array(x)
    y = np.array(y)
    z = abs(np.array(z))
    
    return x, y, z


def actual_event(array, thresh):
    
    ########################## array = z
    actual = []
    
    for i in array:
        actual.append(1 if i >= thresh else 0)
    
    actual = np.array(actual)
    
    return actual


def pred_event_current(array, x, actual, thresh):
    
    predicted = []
    for i in range(len(x)):
        x_pred = x[i][143] - x[i][0]
        x_pred = abs(x_pred)
        
        if x_pred >= thresh:
            ins = 1 ######################### Predicted True
        
        else:
            ins = 0 ######################### Predicted False
        
        predicted.append(ins)
    predicted = np.array(predicted)
    
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR


def pred_event_reg(array, thresh, actual):
    
    '''
    Array = slope of the regression
    '''
    predicted = []
    
    for i in range(len(array)):
        if i >= thresh:
            inst = 1
        else:
            inst = 0
        predicted.append(inst)
    predicted = np.array(predicted)
        
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR


def regression(array):
    
    lin_reg = []
    slope = []
    for i in range(len(array)):
        x = array[i]
        nm = np.arange(0, len(x))
        
        (a_s, b_s, r, tt, stderr) = stats.linregress(nm, x)
        
        reg = polyval([a_s,b_s],nm)
        lin_reg.append(reg)
        slope.append(a_s)
        
    lin_reg = np.array(lin_reg)
    slope = np.array(slope)

    return lin_reg, slope




def confu_matrix(array, thresh):
    
    x = []
    y = []
    z = []
    
    m = 144 ##### 3 day data
    t = 145 ##### for extension
    
    for i in range(len(array) - t):
        
        x.append(array[i:i+m])
        y.append(array[i+m])
        z.append(array[i+t] - array[i+m]) ################### Check if the next time data moves
        
    x = np.array(x)
    y = np.array(y)
    z = abs(np.array(z))
    
    ######################################################################## ACTUAL
    actual = []
    
    for i in z:
        actual.append(1 if i >= thresh else 0)
    
    actual = np.array(actual)
    
    ######################################################################## PREDICTED
    #dif = []
    predicted = []
    for i in range(len(x)):
        x_pred = x[i][143] - x[i][0]
        x_pred = abs(x_pred)
        
        if x_pred >= thresh:
            ins = 1 ################ Predicted true
            
        else:
            ins = 0 ################ Predicted false
        
        predicted.append(ins)
        
    predicted = np.array(predicted)
    
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR


for j in threshold:
        t = j
        for i in cols:
            node = data[data.node_id == i]
            if node.empty:
                i += 1
                
            else:
                xz = node.xz
                ma,sg = fitting(xz)
                try:
                    x,y,z = get_features(sg)
                    actual = actual_event(z, t)################################ True event
                    current_TPR, current_FPR = current_prediction(x,actual,t)
                    regr, slope = regression(x,t)
                    regression_tpr, regression_fpr = reg_prediction(slope, t, actual)
                    #TPR,FPR = confu_matrix(sg, t)
                    cur_tpr.append(current_TPR)
                    cur_fpr.append(current_FPR)
                    reg_tpr.append(regression_tpr)
                    reg_fpr.append(regression_fpr)
                    n_id.append(node.node_id.max())
                    thresh.append(t)
                except:
                    pass
                i += 1
        
df_current = pd.DataFrame({'id':n_id, 'tpr':cur_tpr, 'fpr':cur_fpr, 'thresh':thresh})
#df_current = df_current[['id', 'tpr', 'fpr', 'thresh']]
#df_current = df_current.fillna(0)


df_reg = pd.DataFrame({'id':n_id, 'tpr':reg_tpr, 'fpr':reg_fpr, 'thresh':thresh})
#df_reg = df_reg[['id', 'tpr', 'fpr', 'thresh']]
#df_reg = df_reg.fillna(0)
