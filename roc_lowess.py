# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:12:57 2019
"""

import matplotlib.pyplot as plt
import filterdata as filt
import pandas as pd
import numpy as np
import mysql.connector as sql
from datetime import datetime, date, time, timedelta
from scipy import stats, polyval
from sklearn.metrics import confusion_matrix
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.dates as md
import fitting
from pyfinance.ols import PandasRollingOLS

start = '2018-01-02'
end = '2018-06-30'
sensor = 'tilt_dadtb'
seg_len = 1.5



#db_connection = sql.connect(host='192.168.150.75', database='senslopedb', 
#                            user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

db_connection = sql.connect(host='127.0.0.1', database='senslopedb', 
                            user='root', password='senslope')

def data(start, end, sensor):
    read = db_connection.cursor()
    query = "SELECT * FROM senslopedb.%s" %(sensor)
    query += " WHERE ts BETWEEN '%s' AND '%s'" %(start, end)
      
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names

    d.drop(['batt'], axis=1, inplace=True)
    d = d[d.type_num == 32]
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

def node_inst_vel(filled_smoothened, roll_window_numpts, start):
    print (filled_smoothened)
    try:          
        lr_xz = PandasRollingOLS(y=filled_smoothened.xz, x=filled_smoothened.td,
                    window=roll_window_numpts)
    
    except:
        print('Error')
        pass
    return lr_xz

def get_features(array):
    
    x = []
    y = []
    z = []
    
    m = 144 ##### 3 day data (144), 2 day data (96)
    t = m+1 ##### for extension
    
    for i in range(len(array) - t):
        
        x.append(array[i:i+m])############################### Gets 3 day data
        y.append(array[i+m]) 
        z.append(array[i+t] - array[i+m]) ################### Calculate displacement of the next data after 3 days
        
    x = np.array(x)
    y = np.array(y)
    z = abs(np.array(z))
    
    return x, y, z



def get_rt_window(rt_window_length,roll_window_size,num_roll_window_ops,endpt):
    
    ##DESCRIPTION:
    ##returns the time interval for real-time monitoring,

    ##INPUT:
    ##rt_window_length; float; length of real-time monitoring window in days
    ##roll_window_size; integer; number of data points to cover in moving window operations
    
    ##OUTPUT: 
    ##end, start, offsetstart; datetimes; dates for the end, start and offset-start of the real-time monitoring window 

    ##round down current time to the nearest HH:00 or HH:30 time value
    
    end_Year=endpt.year
    end_month=endpt.month
    end_day=endpt.day
    end_hour=endpt.hour
    end_minute=endpt.minute
    if end_minute<30:end_minute=0
    else:end_minute=30
    end=datetime.combine(date(end_Year,end_month,end_day),time(end_hour,end_minute,0))

    #starting point of the interval
    start=end-timedelta(days=rt_window_length)
    
    #starting point of interval with offset to account for moving window operations 
    offsetstart=end-timedelta(days=rt_window_length+((num_roll_window_ops*roll_window_size-1)/48.))
    
    return end, start, offsetstart



def fill_smooth(df, offsetstart, end, roll_window_numpts, to_smooth, to_fill):    
    if to_fill:
        # filling NAN values
        df = df.fillna(method = 'pad')
        
        #Checking, resolving and reporting fill process    
        if df.isnull().values.any():
            for n in ['xz', 'xy']:
                if df[n].isnull().values.all():
#                    node NaN all values
                    df[n] = 0
                elif np.isnan(df[n].values[0]):
#                    node NaN 1st value
                    df[n] = df[n].fillna(method='bfill')

    #dropping rows outside monitoring window
    df=df[(df.index >= offsetstart) & (df.index <= end)]
    
    if to_smooth and len(df)>1:
        df = df.rolling(window=roll_window_numpts, min_periods=1).mean()
        df = df[roll_window_numpts-1:]
        return np.round(df, 4)
    else:
        return df
    

def low_ess(y,x,fraction):
    filtered = lowess(y, x, frac = fraction)
    
    return filtered

def current_pred(array,thresh):

    predicted = [1 if i >= thresh else 0 for i in array]

    predicted = np.array(predicted)
    
    return predicted


def acceleration(vel, td, thresh):
    
    accel = PandasRollingOLS(y=pd.Series(vel),x=pd.Series(td),window=7)
    accel = accel.beta.values
    
#    array = array.reshape(len(array))
#    vel = []
#    m = 2
#    for i in range(len(array) - m):
#        vel.append(array[i+m])
#        
#    vel = pd.Series(vel)
#    accel = np.array(vel - vel.shift(1))
#    accel = abs(accel)    
#    actual = [1 if i >= (0.0003) else 0 for i in accel]
#    actual = np.array(actual)
    
    return accel


def confusion_mat(actual, predicted):
        
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR

data = data(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(data, seg_len)


#end,start,offsetstart = get_rt_window(3, 7,1,pd.to_datetime(data.tail(1).ts.values[0]))

#data = fill_smooth(data.set_index('ts'),offsetstart,end,7,0,1)
#data = data.reset_index()

data['td'] = data.ts.values - data.ts.values[0]

data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1,'D'))
n=25
#n = data.node_id.max() + 1 

cols = np.arange(1,n+1)

lws_arrays = []
velocity = []

threshold = np.arange(0.005,0.1,0.002)


tpr = []
fpr = []
n_id = []
thresh = []
fn_actual = []
fn_pred = []
col_pred = []
col_act = []
for t in threshold:
    actual = []
    predicted = []
    for i in cols:
        node = data[data.node_id == i]
    
        if len(node)<2000:
            i += 1
        else:
            
            try:
                #node = filt.apply_filters(node)
                xz=np.array(node.xz)
                timestamp = np.array(node.ts.values)
                df_td = np.array(node.td.values)
                
                rol_xz = fitting.rolling_window(xz, window=17)
                timestamp = fitting.rolling_window(timestamp, window=17)
                df_td = fitting.rolling_window(df_td, window=17)
                
                lws_xz, lst_val, ts, n_td = fitting.rolling_lws(rol_xz, timestamp, df_td)
                lst_val = np.array(lst_val)
#                
#                lws = low_ess(xz,np.arange(len(xz)), fraction = 0.1)
#                lws = lws[:,1]
#                
#                lws = pd.Series(lws)
                
                vel = PandasRollingOLS(y=pd.Series(lst_val),x=pd.Series(lst_val),window=7)
                vel = abs(vel.beta.values)
                vel = vel[0:2000]
                td = df_td[0:2000]
#                vel = PandasRollingOLS(y=lws, x=node.td,window=7)
#                vel = abs(vel.beta.values)
#                
#                vel = vel[0:20000]
#                
                accel = acceleration(vel,td,t)
#                pred = current_pred(vel,t)
#                
#                start_index = len(pred) - len(act)
#            
#                actual.append(act)
#                predicted.append(pred[start_index:])
            except:
                pass
        
#    col_act.append(list(map(sum,np.transpose(actual))))
#    col_pred.append(list(map(sum,np.transpose(predicted))))
#    thresh.append(t)
#    print ('##### thresh = %.6f #####'%(t))
#    print ('Actual', len(actual))
#    print ('Predicted', len(predicted))
#            
#thresh = np.array(thresh)
#col_act = np.array(col_act)
#col_pred = np.array(col_pred)        
#
#for j in range(len(col_act)):
#    fn_actual.append([1 if i!=0 else 0 for i in col_act[j]])
#    fn_pred.append([1 if i!=0 else 0 for i in col_pred[j]])
#
#fn_actual = np.array(fn_actual)
#fn_pred = np.array(fn_pred)
#
#for m in range(len(fn_actual)):
#    
#    try:
#        
#        current_TPR, current_FPR = confusion_mat(fn_actual[m], fn_pred[m])
#        
#        tpr.append(current_TPR)
#        fpr.append(current_FPR)
#
#    except:
#        tpr.append(0)
#        fpr.append(0)
#        print('Error in', m)
#        
#        pass
#tpr = np.array(tpr)
#
#fpr = np.array(fpr)
#
#df = pd.DataFrame({'tpr':tpr, 'fpr':fpr, 'thresh':thresh})
#df = df.fillna(0)
#adnal1 = pd.DataFrame({'tpr':1, 'fpr':1, 'thresh':1},index=[0])
#adnal2 = pd.DataFrame({'tpr':0, 'fpr':0, 'thresh':0},index=[0])
#
#final = pd.concat([df, adnal1])
#final = pd.concat([final, adnal2])
#final = final.sort_values(['fpr'])
#final = final[['tpr', 'fpr', 'thresh']]
#
#
#plt.plot(final.fpr,final.tpr, color = 'blue', marker='o')
#plt.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
#plt.title('ROC Plot ({})'.format(sensor), fontsize = 30)
#plt.rcParams['xtick.labelsize']=20
#plt.rcParams['ytick.labelsize']=20
#plt.ylabel('TPR', fontsize = 25)
#plt.xlabel('FPR', fontsize = 25)
############################################## finding distance from perfect
###distance =[]
###
###for i in np.arange(0, len(final)):
###    x = np.array(final.fpr)
###    y = np.array(final.tpr)
###    dist = math.sqrt(((0-x[i])**2)+((1-y[i]))**2)
###    distance.append(dist)
###x2 = 0
###y2 = 1
###
###x1 = np.array(final.fpr)
###y1 = np.array(final.tpr)
###
###min_dist = np.where(distance == distance.min())
###plt.plot([x1[min_dist], 0],[1,y1[min_dist]], linestyle = '-.',marker='o')
##
##
