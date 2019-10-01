# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:22:54 2019
"""
from datetime import timedelta
import numpy as np
import pandas as pd
import mysql.connector as sql
import filterdata as filt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

start = '2017-01-28'
end = '2017-09-30'
sensor = 'tilt_nagsa'
seg_len = 1.5


def data(start, end, sensor):
    read = db_connection.cursor()
    query = "SELECT * FROM senslopedb.%s" %(sensor)
    query += " WHERE ts BETWEEN '%s' AND '%s'" %(start, end)
      
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names

    d.drop(['batt'], axis=1, inplace=True)
    d = d[d.type_num == 11]
    return d


def accel_to_lin_xz_xy(data, seg_len):

    #DESCRIPTION
    #converts accelerometer data (xa,ya,za) to corresponding tilt expressed as
    #horizontal linear displacements values, (xz, xy)
    
    #INPUTS
    #seg_len; float; length of individual column segment
    #xa,ya,za; array of integers; accelerometer data (ideally, -1024 to 1024)
    
    #OUTPUTS
    #xz, xy; array of floats; horizontal linear displacements along the planes 
    #defined by xa-za and xa-ya, respectively; units similar to seg_len
    
    xa = data.xval.values
    ya = data.yval.values
    za = data.zval.values

    theta_xz = np.arctan(za / (np.sqrt(xa**2 + ya**2)))
    theta_xy = np.arctan(ya / (np.sqrt(xa**2 + za**2)))
    xz = seg_len * np.sin(theta_xz)
    xy = seg_len * np.sin(theta_xy)
    
    data['xz'] = np.round(xz,4)
    data['xy'] = np.round(xy,4)
    
    return data


data = data(start, end, sensor)
#data = accel_to_lin_xz_xy(data, seg_len)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(filtered, seg_len)


n = data.node_id.max() + 1 

#cols = np.arange(1,n)
#fig = plt.figure()
#fig.subplots_adjust(hspace=0)
#for i in cols:
#    ax = fig.add_subplot(len(cols), 1, i)
#    x = data[data.node_id == i]
#    ax.plot(x.ts,x.xz, label = '{}'.format(i))
#    ax.yaxis.set_visible(False)
#    plt.legend(loc='right')
#    i += 1
#fig.suptitle('{}'.format(sensor), fontsize = 25)
#plt.show()


############################
#fig, axs = plt.subplots(nrows=nplots, ncols=1, sharex=True)
#for i, alpha in enumerate(vals):
#    axs[i].plot(np.linspace(0,1,100), alpha * np.linspace(0,1,100)**2)
#plt.show()

########################################################################### ARIMA
def test_stationarity(series):
    dftest = adfuller(series, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value',
                        '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
    
    return dfoutput

node = data[data.node_id == 10]
xz = node.xval
ts = node.ts
###############################################################################
#Determining rolling statistics
rolmean = xz.rolling(12).mean()
rolstd = xz.rolling(12).std()

##Plot rolling statistics:
#orig = plt.plot(xz, color = 'blue', label = 'Original')
#mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
#std = plt.plot(rolstd, color = 'green', label = 'Rolling Std')
#
#plt.legend(loc = 'best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.show(block=False)


################################################################################1. Eliminate trend

### Moving average
ts_log = node.xz
moving_avg = ts_log.rolling(12).mean()


ts_log_moving_avg_diff = ts_log - moving_avg ###subtract moving ave to original data
ts_log_moving_avg_diff.dropna(inplace=True)

#stationarity = test_stationarity(ts_log_moving_avg_diff)


decomposition = seasonal_decompose(ts_log, freq = 15,model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid




plt.subplot(411)
plt.plot(ts,ts_log, label='Original')
plt.legend(loc='right')
plt.subplot(412)
plt.plot(ts,trend, label='Trend')
plt.legend(loc='right')
plt.subplot(413)
plt.plot(ts,seasonal,label='Seasonality')
plt.legend(loc='right')
plt.subplot(414)
plt.plot(ts,residual, label='Residuals')
plt.legend(loc='right')
plt.suptitle('Node {}'.format(node.node_id.max()), fontsize = 25)
#######################################################################

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)


lag_acf = acf(ts_log_diff, nlags = 20)
lag_pacf = pacf(ts_log_diff, nlags = 20, method = 'ols') ### ols - regression of time series on lags of it and on constant
# Plot ACF:
plt.subplot(121)
bins = np.arange(len(lag_acf))
plt.bar(bins,lag_acf)
plt.axhline(y=0, linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation Function')

# Plot PACF:
plt.subplot(122)
bins2 = np.arange(len(lag_pacf))
plt.bar(bins2,lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#


############################################################################### ARIMA model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

############################################################################### Taking it back to original scale

#predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
#
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#
#
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#
#predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.plot(ts)
#plt.plot(predictions_ARIMA)
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

#
#plt.subplot(121)
#plt.plot(ts_log_diff)
#plt.subplot(122)
#plt.plot(results_ARIMA.fittedvalues)



plt.plot(ts,trend, label='Trend')
plt.axhline(y = trend.mean(), color = 'r')
