# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 06:09:45 2019
"""

import matplotlib.pyplot as plt
import filterdata as filt
import pandas as pd
import numpy as np
import mysql.connector as sql
from datetime import datetime, date, time, timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.dates as md
from pyfinance.ols import PandasRollingOLS
#import rolling
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

dyna_colors = [(22,82,109),(153,27,30),(248,153,29)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(dyna_colors)):
    r_d,g_d,b_d = dyna_colors[i]
    dyna_colors[i] = (r_d / 255., g_d / 255., b_d / 255.)


#db_connection = sql.connect(host='192.168.150.75', database='senslopedb', 
#                            user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')


db_connection = sql.connect(host='127.0.0.1', database='senslopedb', 
                            user='root', password='senslope')

def data_query(start, end, sensor):
    read = db_connection.cursor()
    query = "SELECT * FROM senslopedb.%s" %(sensor)
    query += " WHERE ts BETWEEN '%s' AND '%s'" %(start, end)
      
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names
    d.drop(['batt'], axis=1, inplace=True)
    try:
        nm = min(d.type_num)
        d = d[d.type_num == 32]
        print(nm)

    except:
        print ('version 1')
        d.dropna()
        pass

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



def get_rt_window(rt_window_length,roll_window_size,num_roll_window_ops,endpt):
    
    ##DESCRIPTION:
    ##returns the time interval for real-time monitoring,

    ##INPUT:
    ##rt_window_length; float; length of real-time monitoring window in days
    ##roll_window_size; integer; number of data points to cover in moving window operations
    
    ##OUTPUT: 
    ##end, start, offsetstart; datetims; dates for the end, start and offset-start of the real-time monitoring window 

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

def moving_ave(array):

    data = pd.DataFrame({'x':array})
    
    ma = data.rolling(7).mean()
    ma = np.array(ma)
    
    return ma


def low_ess(y,x,fraction):
    lws = lowess(y, x, frac = fraction)
    
    return lws

#############
def rolling_lws(df,slope,window):
    
    if slope == 'xz':        
        sl = df.xz
    else:
        sl = df.xy
    td = df.td
    ts = df.ts
    
    m = window ###embedding dimension
    
    lws = []
    last_val=[]
    t_d = []
    t_s = []
    for i in  range(len(sl) - m):
        x = sl[i:i+m]
        
        fit_a = low_ess(np.array(x), np.arange(len(x)), fraction = 1)
        fit_a = fit_a[:,1]
        
        lws.append(fit_a)
        last_val.append(fit_a[-3])
##############################################################################td
        y = np.array(td[i:i+m])
        t_d.append(y[-1])
##############################################################################ts
        t = np.array(ts[i:i+m])
        t_s.append(t[-1])
        
        
    last_val = np.array(last_val)
    t_d = np.array(t_d)
    t_s = np.array(t_s)
    
    return lws,last_val,t_d, t_s

def lowess_fitting(df):
    
    x_z = df.xz
    x_y = df.xy
    td = df.td
    ts = df.ts
    
    m = 17 ###embedding dimension
    
    lws_xz = []
    lws_xy = []
    
    last_val_xz=[]
    last_val_xy=[]
    
    t_d = []
    t_s = []
    
    d_f = pd.DataFrame()
    for i in  range(len(x_z) - m):
        xz = x_z[i:i+m]
        
        fit_xz = low_ess(np.array(xz), np.arange(len(xz)), fraction = 1)
        fit_xz = fit_xz[:,1]
        
        lws_xz.append(fit_xz)
        last_val_xz.append(fit_xz[-3])
        
        
        xy = x_y[i:i+m]
        
        fit_xy = low_ess(np.array(xy), np.arange(len(xy)), fraction = 1)
        fit_xy = fit_xy[:,1]
        
        lws_xy.append(fit_xy)
        last_val_xy.append(fit_xy[-3])
##############################################################################td
        y = np.array(td[i:i+m])
        t_d.append(y[-1])
##############################################################################ts
        t = np.array(ts[i:i+m])
        t_s.append(t[-1])
        
        daf = pd.DataFrame({'ts':t_s,'td':t_d, 'lws_xz':last_val_xz, 'lws_xy':last_val_xy})
        
        
        d_f = pd.concat([d_f, daf])
        
    
    return d_f

def velocity(df):
   
    vel_xz = PandasRollingOLS(y=df.lws_xz,x=df.td, window=7)
    df['vel_xz'] = ([np.nan] * 6) + list(abs(vel_xz.beta.values))
    
    vel_xy = PandasRollingOLS(y=df.lws_xy,x=df.td, window=7)
    df['vel_xy'] = ([np.nan] * 6) + list(abs(vel_xy.beta.values))
    
    return df


#def velocity2(df):
#    
#    try:
#        vel_ = PandasRollingOLS(y=df.lws_xz,x=df.td, window=7)
#        df['vel_xz'] = ([np.nan] * 6) + list(abs(vel_.beta.values))
#    except:
#        vel_ = PandasRollingOLS(y=df.td,x=df.lws_xz, window=7)
#        df['vel_xz'] = ([np.nan] * 6) + list(abs(vel_.beta.values))
#
#    try:
#        vel_ = PandasRollingOLS(y=df.lws_xy,x=df.td, window=7)
#        df['vel_xy'] = ([np.nan] * 6) + list(abs(vel_.beta.values))
#    except:
#        vel_ = PandasRollingOLS(y=df.td,x=df.lws_xy, window=7)
#        df['vel_xy'] = ([np.nan] * 6) + list(abs(vel_.beta.values))        
#
#    return df



def q_sensors():
    read = db_connection.cursor()
    query = "SELECT tsm_name FROM tsm_sensors"   
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names
    
    sen = np.array(d.tsm_name)

    return sen


sensors = 'tilt_'+ q_sensors()
df = pd.read_csv('percent_movement.csv')
df['ts'] = pd.to_datetime(df['ts']).astype('datetime64[D]')    


tsm = ['magta']
print(tsm)
s_final = pd.DataFrame()
errors = []
for n in tsm:
    print('on = ', n)
    
    try:
#        tsm_name = n
        sensor = 'tilt_' + n
        seg_len = 1.5
        
        ##############################################################################
        
        sen = df[df.tsm_name == n]
        
        node_triggers = np.unique(sen.node_id)
        ts_unique = np.unique(sen.ts).astype('datetime64[D]')
#        print(node_triggers)
#        print(ts_unique)
        ##############################################################################
        final = pd.DataFrame()
        for i in ts_unique:
            start = i - np.timedelta64(20, 'D')
            end = i + np.timedelta64(10, 'D')
            data = data_query(start, end, sensor)
            
            filtered = filt.apply_filters(data)
            data = accel_to_lin_xz_xy(data, seg_len)
            
            data['td'] = data.ts.values - data.ts.values[0]
            data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1, 'D'))
            
            emp = pd.DataFrame()
            uni = sen[sen.ts == i]
            nodes = np.unique(uni.node_id)
            print(i)
            print (nodes)
            for j in nodes:
                try:
                    node = data[data.node_id == j]
            
                    lws_xz,last_val_xz,t_d, t_s = rolling_lws(node, slope = 'xz', window = 17)
                    lws_xy,last_val_xy,t_d, t_s = rolling_lws(node, slope = 'xy', window = 17)
                    
                    daf = pd.DataFrame({'o_ts':i,'ts':t_s,'td':t_d, 'lws_xz':last_val_xz, 'lws_xy':last_val_xy, 'node_id':j, 'tsm_name':n})
                    vel_ = velocity(daf)
                    emp = pd.concat([emp, vel_])
                    emp['prediction'] = np.where((emp['vel_xz'] >= 0.032)|(abs(emp['vel_xy']) >= 0.032), 1, 0)
                except:
                    print('ip_node = ', j)
                pass
            
            final = pd.concat([final, emp])
        s_final = pd.concat([s_final, final])
    except:
        print('p = ', n)
        errors.append(n)
        pass
    
    
s_final.to_csv('magta.csv')















#data = data_query(start, end, sensor)
#filtered = filt.apply_filters(data)
#data = accel_to_lin_xz_xy(data, seg_len)
#
#data['td'] = data.ts.values - data.ts.values[0]
#
#data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1,'D'))
#
#nodes = np.array([14,15, 11, 19, 22, 23, 17, 21,  6,  1,  3,  4,  5,  7,  8,  9, 10, 12, 13, 16, 18, 20, 24, 25])
#
#emp = pd.DataFrame()
#
#for i in nodes:
##try:
#    
#    node = data[data.node_id == i]
#    
#    lws_xz,last_val_xz,t_d, t_s = rolling_lws(node, slope = 'xz', window = 7)
#    lws_xy,last_val_xy,t_d, t_s = rolling_lws(node, slope = 'xy', window = 7)
#    
#    daf = pd.DataFrame({'ts':t_s,'td':t_d, 'lws_xz':last_val_xz, 'lws_xy':last_val_xy, 'node_id':i})
#    vel_ = velocity(daf)
#    emp = pd.concat([emp, vel_])
#    emp['prediction'] = np.where((emp['vel_xz'] >= 0.032)|(abs(emp['vel_xy']) >= 0.032), 1, 0)
##except:
##    print('error in', i)
##    pass
#
#
#
#plt.plot(emp.ts,emp.prediction)
#
##emp.to_csv('{}.csv'.format(sensor))