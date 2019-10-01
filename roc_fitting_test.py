# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:33:25 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import fitting as fit
from sklearn.metrics import confusion_matrix




x = np.linspace(-2, 4, 2000)
#y = (x+3) * ((x-2)**2) * (x+1)**3
y = (5 * x**5) - (20*(x**4)) + (5 * (x**3)) + (50 *(x**2) - (20*x)) -40
mean_noise = 0
noise = np.random.normal(mean_noise, 30, len(y))

y_noise = y + noise

#plt.plot(x,y_noise, label = 'Noise')
#plt.plot(x,y,color ='red', label = 'Actual')
#plt.title(r'$5x^5 - 20x^4 + 5x^3 + 50x^2 - 20x - 40$',
#                    fontsize = 30)
#plt.legend(loc='upper right')


def displacement(x,y):
    data = pd.DataFrame({'x':x, 'y':y})
    
    data['disp'] = data.y - data.y.shift(2)
    
    return data


def score(x_in,thresh):
    score = [1 if x>=thresh else 0 for x in x_in]
    return score

############################################################################### actual
data = displacement(x,y)
disp = data.disp

#data['actual'] = np.array([1 if x>=1 else 0 for x in data.disp])
############################################################################### fitting
### Moving average
ma = fit.moving_ave(y_noise)
ma = ma.reshape(len(ma))
data_ma = displacement(x,ma)

### SG
sg = fit.savitzky_golay(y_noise)
data_sg = displacement(x,sg)

### Loess
loess = fit.low_ess(y_noise, x, fraction = 0.05)
data_loess = displacement(x, loess[:,1])

### EMA
ema = fit.ema(y_noise)
ema = np.array(ema.x)
data_ema = displacement(x, ema)

### ann
ann = fit.ann(y_noise)
dn_ann = (ann) * (max(y_noise) - min(y_noise)) + min(y_noise)
start_index = len(x) - (len(dn_ann))

data_ann = displacement(x[start_index:], dn_ann)
#data_ann['score'] = np.array([1 if x>=1 else 0 for x in data_ann.disp])

def get_features(array, train_ratio = 0.75):
    
    x = []
    y = []

    m = 3 
    
    for i in range(len(array) - m):
        
        x.append(array[i:i+m])############################### Gets 3 day data
        y.append(array[i+m]) 
        
        
    x = np.array(x)
    y = np.array(y)
    
    last_index = int(len(x)*train_ratio)
    
    train_x = x[0:last_index]
    train_y = y[0:last_index]
    test_x = x[last_index:]
    test_y = y[last_index:]
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = get_features(y_noise)

def clf(in_put, out_put):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LassoLarsCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import LinearSVR
    from tpot.builtins import StackingEstimator, ZeroCount

    
    exported_pipeline = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=LinearSVR(C=15.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=1e-05)),
            PCA(iterated_power=7, svd_solver="randomized"),
            ZeroCount(),
            LassoLarsCV(normalize=True)
            )

    exported_pipeline.fit(in_put, out_put)
    results = exported_pipeline.predict(in_put)
    
    return results

pred_train = clf(train_x, train_y)
pred_test = clf(test_x, test_y)

final = np.concatenate((pred_train,pred_test))
############################################################################### ROC


def con_mat(actual,fitted):
    
    thresh = np.arange(1,11,0.01)
    
    data = pd.DataFrame()
    
    for i in thresh:
        try:
            true = [1 if x>=i else 0 for x in actual]
            pred = [1 if x>=i else 0 for x in fitted]
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            
            TPR = (tp / (tp + fn))
            FPR = (fp / (fp + tn))
            
            d = pd.DataFrame({'TPR':[TPR], 'FPR':[FPR], 'thresh':[i]})
            data =pd.concat([data,d])
        except:
            print('ERROR on {}'.format(i))
            pass
        
    adnal1 = pd.DataFrame({'TPR':[1], 'FPR':[1], 'thresh':[100]})
    adnal2 = pd.DataFrame({'TPR':[0], 'FPR':[0], 'thresh':[0]})
    data = pd.concat([adnal1,data])
    data = pd.concat([adnal2,data])
    return data

plt.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black', label ='random')
d_ma = con_mat(data.disp, data_ma.disp)
d_ma = d_ma.fillna(0)
d_ma = d_ma.sort_values(by=['FPR'])
plt.plot(d_ma.FPR,d_ma.TPR, color = 'green', label = 'MA')
print('AUC MA =',metrics.auc(d_ma.FPR,d_ma.TPR))


d_sg = con_mat(data.disp, data_sg.disp)
d_sg = d_sg.fillna(0)
d_sg = d_sg.sort_values(by=['FPR'])
plt.plot(d_sg.FPR,d_sg.TPR, color = 'blue', label = 'SG')
print('AUC SG = ',metrics.auc(d_sg.FPR,d_sg.TPR))

d_lo = con_mat(data.disp, data_loess.disp)
d_lo = d_lo.fillna(0)
d_lo = d_lo.sort_values(by=['FPR'])
plt.plot(d_lo.FPR,d_lo.TPR, color = 'red', label = 'LOWESS')
print('AUC lowess =',metrics.auc(d_lo.FPR,d_lo.TPR))

d_ema = con_mat(data.disp, data_ema.disp)
d_ema = d_ema.fillna(0)
d_ema = d_ema.sort_values(by=['FPR'])
plt.plot(d_ema.FPR,d_ema.TPR, color = 'orange', label = 'EMA')
print('AUC ema =',metrics.auc(d_ema.FPR,d_ema.TPR))

d_ann = con_mat(data.disp[start_index:], data_ann.disp)
d_ann = d_ann.fillna(0)
d_ann = d_ann.sort_values(by=['FPR'])
plt.plot(d_ann.FPR,d_ann.TPR, color = 'hotpink', label = 'ANN')
print('AUC ann =',metrics.auc(d_ann.FPR,d_ann.TPR))

d_tpot = con_mat(data.disp[start_index:],final)
d_tpot = d_tpot.fillna(0)
d_tpot = d_tpot.sort_values(by=['FPR'])
plt.plot(d_tpot.FPR, d_tpot.TPR, color = 'maroon', label = 'LSTM')
print('AUC tpot =',metrics.auc(d_tpot.FPR,d_tpot.TPR))

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize = 15)
plt.xlabel('FPR', fontsize = 25)
plt.ylabel('TPR', fontsize = 25)
plt.title('ROC plots', fontsize = 25)