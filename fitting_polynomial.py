# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:26:03 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import fitting as fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR



x1 = np.linspace(-4, 4, 1000)
x2 = np.linspace(4, 8, 1000)
y1 = (8 * x1**5) - (20*(x1**4)) + (5 * (x1**3)) + (50 *(x1**2) - (20*x1)) -40
y2 = -2*x1**4 + 4*x1**3
y2 = -y1-10000
#y2 = 10*np.sin(x1/(2*np.pi))

mean_noise = 20
mean_noise2 = 50
noise1 = np.random.normal(mean_noise, 200, len(y1))
noise2 = np.random.normal(mean_noise2, 550, len(y2))

y1_noise = y1 + noise1
y2_noise = -y1-10000 + noise2

x = np.concatenate((x1,x2))
y = np.concatenate((y1_noise,y2_noise))

ma = fit.moving_ave(y)
sg = fit.savitzky_golay(y)
ema = fit.ema(y)
lws = fit.low_ess(y,x,fraction=0.01)
ann = fit.ann(y)
dn_ann = (ann) * (max(y) - min(y)) + min(y)
start_index = len(x) - (len(dn_ann))



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

train_x, train_y, test_x, test_y = get_features(y)

#tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2)
#tpot.fit(train_x, train_y)
#print(tpot.score(test_x, test_y))
#tpot.export('polynomial5.py')


# Average CV score on the training set was:-1266.7900678552883

def clf(in_put, out_put):
    
    exported_pipeline = make_pipeline(
                        MaxAbsScaler(),
                        LinearSVR(C=15.0, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.1)
                        )   

    exported_pipeline.fit(in_put, out_put)
    results = exported_pipeline.predict(in_put)
    
    return results

pred_train = clf(train_x, train_y)
pred_test = clf(test_x, test_y)


final = np.concatenate((pred_train,pred_test))
noise = np.concatenate((train_x[:,0],test_x[:,0]))
start_index2 = len(x) - (len(final))




fig, (axes) = plt.subplots(7, 1, sharex = True)
fig.set_size_inches(18.5, 10.5, forward=True)
#axes[0].set_title(r'$5x^5 - 20x^4 + 5x^3 + 50x^2 - 20x - 40$',
#                    fontsize = 30, y=1.2)

axes[0].plot(x,y, alpha = 0.50, label ='Noise',color='gray')
axes[0].plot(np.concatenate((x1,x2)),np.concatenate((y1,y2)),color='black', 
        alpha=1,label='actual')
axes[0].legend(loc = 'upper right')

axes[1].plot(x,ma,label='Moving average', color='orangered')
#axes[1].legend(bbox_to_anchor=(1, 1.05), fancybox=True, shadow=True)
axes[1].legend(loc = 'upper right')

axes[2].plot(lws[:, 0], lws[:, 1],color='olive',label='Lowess')
axes[2].legend(loc = 'upper right')
#axes[2].text((x2[0]-1.25), (y2.max()-2), textstr, style='oblique', fontsize=15,
#            bbox={'facecolor':'white', 'alpha':0.5, 'pad':15})

axes[3].plot(x,ema,label='EMA',color='royalblue')
axes[3].legend(loc = 'upper right')
axes[3].set_ylabel('Y', fontsize=13,rotation=0)


axes[4].plot(x,sg,color='orange',label='SG')
axes[4].legend(loc = 'upper right')

axes[5].plot(x[start_index:],dn_ann,color='green',label='ANN (m = 3)')
axes[5].legend(loc = 'upper right')

axes[6].plot(x[start_index2:],final,color='red', label = 'LSTM')
axes[6].legend(loc = 'upper right')
axes[6].set_xlabel('X',fontsize=13)