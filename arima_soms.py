# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 18:39:58 2019
"""

import pandas as pd
import numpy as np
import mysql.connector as sql
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose



start = '2017-04-08'
end = '2017-05-26'
soms = 'soms_laysa'


def test_stationarity(series):
    dftest = adfuller(series, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value',
                        '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
    
    return dfoutput


db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')
db_cursor = db_connection.cursor()
db_cursor.execute("SELECT * FROM senslopedb.%s" %(soms))

d = pd.DataFrame(db_cursor.fetchall())
d.columns = db_cursor.column_names

d = d[d.type_num == 110]
d = d[d.node_id == 4]
d.set_index('ts', inplace = True)


mval = d['mval1']
###############################################################################
#Determining rolling statistics
rolmean = mval.rolling(12).mean()
rolstd = mval.rolling(12).std()

#Plot rolling statistics:
orig = plt.plot(mval, color = 'blue', label = 'Original')
mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
std = plt.plot(rolstd, color = 'green', label = 'Rolling Std')

#plt.legend(loc = 'best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.show(block=False)


################################################################################1. Eliminate trend

### Moving average
ts_log = np.log(mval)
moving_avg = ts_log.rolling(12).mean()

ts_log_moving_avg_diff = ts_log - moving_avg ###subtract moving ave to original data
ts_log_moving_avg_diff.dropna(inplace=True)

#test_stationarity(ts_log_moving_avg_diff)




#decomposition = seasonal_decompose(ts_log, freq = 30)
#trend = decomposition.trend
#seasonal = decomposition.seasonal
#residual = decomposition.resid
#
#
#plt.subplot(411)
#plt.plot(ts_log, label='Original')
#plt.legend(loc='right')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='right')
#plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
#plt.legend(loc='right')
#plt.subplot(414)
#plt.plot(residual, label='Residuals')
#plt.legend(loc='right')
#plt.tight_layout()