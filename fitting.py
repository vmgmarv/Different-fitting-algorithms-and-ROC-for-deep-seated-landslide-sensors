# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:38:27 2019
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from statsmodels.nonparametric.smoothers_lowess import lowess
import rolling


def moving_ave(array):
    '''
    Window size = 12
    '''
    data = pd.DataFrame({'x':array})
    
    ma = data.rolling(7).mean()
    ma = np.array(ma)
    
    return ma


def savitzky_golay(array):
    '''
    window_length = 7
    polyorder = 4
    '''
    
    sg = savgol_filter(array, window_length=11,polyorder=4)
    
    return sg


def ema(array):
    data = pd.DataFrame({'x':array})
    ema_short = data.ewm(span=7, adjust = True).mean()

    return ema_short

############################################################################### ANN
def initialize_vector(array):
    '''
    embedding dimension = 3
    train ratio = 70%
    '''
    ### embedding dimension
    m = 3
    
    ###Normalize
    n = (array - min(array))/(max(array) - min(array))
    
    
    x = []
    y = []
    
    for i in range(len(n)-m):
        x.append(n[i:i+m])
        y.append(n[i+m])
    
    x = np.array(x)
    y = np.array(y)
    
    ### Initialize train_test set
    train_ratio = 0.70
    last_index = int(len(x)*train_ratio)
    
    train_x = x[0:last_index]
    train_y = y[0:last_index]
    test_x = x[last_index:]
    test_y = y[last_index:]
    
    return train_x, train_y, test_x, test_y
    

def GetOptimalCLF(train_x, train_y):
    '''
    iteration = 8
    '''
    rand_start = 8
    
    n_input = train_x.shape[1]
    min_loss = 1e10
    
    for i in range(rand_start):
        #print('Iteration number {}'.format(i+1))
        
        #### ANN model
        clf = MLPRegressor(hidden_layer_sizes = (int(round(2*np.sqrt(n_input),0)),2), 
                           activation = 'tanh',solver = 'sgd', 
                           learning_rate = 'adaptive', max_iter = 100000000,tol = 1e-10,
                           early_stopping = True, validation_fraction = 1/3.)
        
        clf.fit(train_x,train_y)
        
        cur_loss = clf.loss_
        
        if cur_loss < min_loss:
            
            min_loss = cur_loss
            max_clf = clf
    
    return max_clf


############################################################################### ANN main
def ann(array):
    train_x, train_y, test_x, test_y = initialize_vector(array)
    clf = GetOptimalCLF(train_x,train_y)
    
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    x = np.concatenate((pred_train,pred_test))
    
    return x

############################################################################### LOWESS (3hr window)
def low_ess(y,x,fraction):
    lws = lowess(y, x, frac = fraction)
    
    return lws

#############
def rolling_window(array,window):
    
    f_list = rolling.Apply(array, window, operation = list)
    f_list = list(f_list)    
    
    return f_list

############ fitting with LOWESS
def rolling_lws(val, timestamp, df_td):
    lws_val = []
    last_val = []
    ts = []
    td = []
    for i in range(len(val)):
        fit_a = low_ess(np.array(val[i]), np.arange(len(val[i])), fraction = 1.1)
        fit_a = fit_a[:,1]
        
        lws_val.append(fit_a)
        end=len(lws_val[i]) - 1
        last_val.append(fit_a[end])
        
    for j in range(len(timestamp)):
        end = len(timestamp[j])-1
        ts.append(timestamp[j][end])
    
    for k in range(len(df_td)):
        end = len(df_td[k])-1
        td.append(df_td[k][end])
    return lws_val, last_val, ts, td
############################################################################### LSTM

### convert an array of values into a dataset matrix
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
    model.add(LSTM(10, input_shape=(1, look_back)))
    model.add(Dropout(2))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_logarithmic_error', 
                  optimizer = 'adam')
    model.fit(trainX, trainY, epochs = 20, 
              batch_size = 1, verbose = 0)
    
    return model



def predict(model, trainX, testX):
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    return trainPredict, testPredict



def lstm_pred(array):
    '''
    Train ratio = 75%
    Look back = 1
    '''

    data = pd.DataFrame({'x':array})


    data_raw = data.values.astype("float32")
    scaler = MinMaxScaler(feature_range = (0, 1))
    
    dataset = scaler.fit_transform(data_raw)

    train_ratio = 0.75

    train_size = int(len(dataset) * train_ratio)

    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("No. of entries (training set, test set): " + str((len(train), len(test))))

    ### Look back to previous data point
    look_back = 1

    ### Reshape into x=t, y=t+1
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


############################################################################### Main LSTM
def lstm_main(array):

    trainPredictPlot, testPredictPlot, trainX, testX = lstm_pred(array)

    testPredictPlot = testPredictPlot[~np.isnan(testPredictPlot)]
    trainPredictPlot = trainPredictPlot[~np.isnan(trainPredictPlot)]

    dn = np.concatenate((trainPredictPlot, testPredictPlot))

    return dn


    