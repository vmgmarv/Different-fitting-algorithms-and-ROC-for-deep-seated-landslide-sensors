# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:07:33 2019
"""

import subsurface_plotting as db
import filterdata as filt
import math
from keras.layers import Dropout  
import pandas as pd
import mysql.connector as sql
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


db_connection = sql.connect(host='192.168.150.75', database='senslopedb',
                            user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

start = '2018-01-01'
end = '2018-12-30'
sensor = 'tilt_sagta'
seg_len = 1.5


data = db.data(start, end, sensor)
filtered = filt.apply_filters(data)
data = db.accel_to_lin_xz_xy(filtered, seg_len)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=2):

	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def lstm_clf(trainX, trainY, look_back):
    
    '''
    LSTM model:
    
        10 neurons
        0.5 Dropout
        Dense????
    '''
    
    model = Sequential()
    model.reset_states()
    model.add(LSTM(2, input_shape=(1, look_back)))
    model.add(Dropout(2))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = 'sgd')
    model.fit(trainX, trainY, epochs = 20, batch_size = 1, verbose = 0)
    
    return model


def predict(model, trainX, testX):
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    return trainPredict, testPredict




def lstm_pred(data):
    col = ['xz']
    data = data[col]
    
    data_raw = data.values.astype("float32")
    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(data_raw)
    
    TRAIN_SIZE = 0.67
    
    train_size = int(len(dataset) * TRAIN_SIZE)
#    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("Number of entries (training set, test set): " + str((len(train), len(test))))
    
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    
    
    lstm_model = lstm_clf(trainX, trainY, look_back)
    
    trainPredict, testPredict = predict(lstm_model, trainX, testX)
    
    
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.8f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.8f RMSE' % (testScore))
    
    
    
    
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    
    #return trainX, testX
    trainX = np.reshape(trainX, (trainX.shape[0]))
    testX = np.reshape(testX, (testX.shape[0]))
    
    return trainPredictPlot, testPredictPlot, trainX, testX




#n = int(data.node_id.max())
n = 7
cols = np.arange(1,n)

fig, (axes) = plt.subplots(n,1, sharex = True, sharey = False)
fig.subplots_adjust(hspace=0)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible = False)


for i in range (0, n):
    
    n = i + 1
    node = data[data.node_id == n]
    
    print ("Node {}...".format(n))
    
    if node.empty:
        print ('#####No data#####')
        i += 1
        
    else:
#
#node = data[data.node_id == 7]

        trainPredictPlot, testPredictPlot, trainX, testX = lstm_pred(node)


        testPredictPlot = testPredictPlot[~np.isnan(testPredictPlot)]
        trainPredictPlot = trainPredictPlot[~np.isnan(trainPredictPlot)]
        
        start_index = len(node) - (len(trainPredictPlot) + len(testPredictPlot))
        
        dn_actual = np.concatenate((trainX, testX))*(max(node.xz) - min(node.xz)) + min(node.xz)
        
        #axes[i].plot(node.ts.values[start_index:], dn_actual, label = 'Node {} - Measured'.format(n))
        axes[i].plot(node.ts.values[start_index:], np.concatenate((trainPredictPlot, testPredictPlot)), lw = 0.85, label = 'Node {} - Predicted'.format(n))
        axes[i].legend(loc = 'upper right')

axes[i-1].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))

#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#ax.plot(node.ts.values[start_index:],np.concatenate((trainPredictPlot,testPredictPlot)), label = 'Predicted')
##ax.plot(node.ts.values[start_index], np.concatenate((trainX, testX)), label = 'Node {} - Measured'.format(n))
##ax.plot(node.ts.values[start_index:],dataset, label = 'Predicted')
##plt.plot(testPredictPlot)
#ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#plt.legend()
