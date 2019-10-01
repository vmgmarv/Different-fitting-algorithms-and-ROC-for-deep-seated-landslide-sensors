# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:42:19 2019
"""

import subsurface_plotting as db
import filterdata as filt
import pandas as pd
import mysql.connector as sql
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor


db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

start = '2017-02-01'
end = '2018-12-30'
sensor = 'tilt_sagta'
seg_len = 1.5


data = db.data(start, end, sensor)
filtered = filt.apply_filters(data)
data = db.accel_to_lin_xz_xy(filtered, seg_len)



def get_features(array, train_ratio):
    
    x = []
    y = []
    
    m = 3 ##### 3 day data (144), 2 day data (96)
    
    for i in range(len(array) - m):
        
        x.append(array[i:i+m])
        y.append(array[i+m]) 
        
    x = np.array(x)
    y = np.array(y)
    
    last_index = int(len(x) * train_ratio)
    
    
    train_x = x[0:last_index]
    train_y = y[0:last_index]
    test_x = x[last_index:]
    test_y = y[last_index:]
    
    return train_x, train_y, test_x, test_y


def getoptimalclf(train_x, train_y, rand_starts = 8):
    n_input = train_x.shape[1]
    
    min_loss = 1e10
    
    for i in range(rand_starts):
        
        print ("Iteration number {}".format(i+1))
        
        clf = MLPRegressor(hidden_layer_sizes = (int(round(2*np.sqrt(n_input),0)),2), activation = 'relu',solver = 'sgd', 
                           learning_rate = 'adaptive', max_iter = 100000000,tol = 1e-10,
                           early_stopping = True, validation_fraction = 1/3.)
        
        clf.fit(train_x, train_y)
        
        cur_loss = clf.loss_
        
        if cur_loss < min_loss:
            
            min_loss = cur_loss
            

            max_clf = clf
    
    return max_clf


n = int(10) 
cols = np.arange(1,n)


fig, (axes) = plt.subplots(n, 1, sharex = True, sharey= False)
fig.subplots_adjust(hspace=0)
#axes[0].set_title('Surficial and Gravimetric \nSensor {}\n {} to {}'.format(site, start, end), fontsize = 14)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

t =  (n/2) + 0.5

#axes[0].plot(data.ts, data.xz.rolling(12).mean(), color = 'blue')
#axes[0].text(end, 0.46, 'Rain', style='oblique',
#        bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})

for i in range (0, n):
    
    n = i + 1
    node = data[data.node_id == n]
    
    print ("Node {}...".format(n))
    
    if node.empty:
        i += 1
        
    else:
    
        displacement = node.xz.dropna().values
        displacement = (displacement - min(displacement))/(max(displacement) - min(displacement))
        
        
        train_x, train_y, test_x, test_y = get_features(displacement, train_ratio = 0.70)
        
        clf_ = getoptimalclf(train_x, train_y)
        
        pred_train = clf_.predict(train_x)
        pred_test = clf_.predict(test_x)
        
        dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
        dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
        
        start_index = len(node) - (len(train_y) + len(test_y))
        
        
        dn_actual = np.concatenate((train_y, test_y))*(max(node.xz) - min(node.xz)) + min(node.xz)
        dn_predict = np.concatenate((pred_train, pred_test))*(max(node.xz) - min(node.xz)) + min(node.xz)

        axes[i].plot(node.ts.values[start_index:], dn_actual, label = '{} - Measured'.format(n))
        axes[i].plot(node.ts.values[start_index:], dn_predict, lw = 0.85, label = '{} - Fitted'.format(n))
        axes[i].legend(loc = 'upper right')
        
    i += 1
    
    
axes[i-1].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    