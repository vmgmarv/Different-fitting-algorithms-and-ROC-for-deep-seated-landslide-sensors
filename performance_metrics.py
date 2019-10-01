# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:32:17 2019
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


'''
Receiver Operating Characteristic
'''
def current_pred(array,thresh):

    predicted = [1 if i >= thresh else 0 for i in array]

    predicted = np.array(predicted)
    
    return predicted


def actual_event(array, thresh):
    
    array = array.reshape(len(array))
#    m = 2
#    for i in range(len(array) - m):
#        vel.append(array[i+m])
        
    array = pd.Series(array)
    accel = np.array(array - array.shift(1)) 
    actual = [1 if i >= (0) else 0 for i in accel]
    actual = np.array(actual)
    
    return actual


def confusion_mat(actual, predicted):
        
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR