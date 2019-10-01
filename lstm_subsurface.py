# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:37:41 2019
"""

import subsurface_plotting as db
import filterdata as filt
import math
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


db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

start = '2016-01-01'
end = '2018-02-01'
sensor = 'tilt_pngta'
seg_len = 1.5


data = db.data(start, end, sensor)
filtered = filt.apply_filters(data)
data = db.accel_to_lin_xz_xy(filtered, seg_len)


def get_features(array, train_ratio):
    
    x = []
    y = []
    
    m = 3 
    
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



node = data[data.node_id == 7]
col = ['xz']
node = node[col]

data_raw = node.values.astype("float32")
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

TRAIN_SIZE = 0.60

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))
    
    

window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)


train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)



def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    
    model.add(LSTM(4, 
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam")
    model.fit(train_X, 
              train_Y, 
              epochs = 20, 
              batch_size = 1, 
              verbose = 2)
    
    return(model)
    
    
# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)


def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

#rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
#rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
#
#print("Training data score: %.2f RMSE" % rmse_train)
#print("Test data score: %.2f RMSE" % rmse_test)



# make predictions
rmse_train, trainPredict = predict_and_score(model1, train_X, train_Y)
rmse_test, testPredict = predict_and_score(model1, test_X, test_Y)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([train_Y])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([test_Y])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()