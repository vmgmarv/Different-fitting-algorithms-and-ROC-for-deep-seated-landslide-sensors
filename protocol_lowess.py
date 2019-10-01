# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:57:30 2019
"""

import matplotlib.pyplot as plt
import filterdata as filt
import pandas as pd
import numpy as np
import mysql.connector as sql
from datetime import datetime, date, time, timedelta
from sklearn.metrics import confusion_matrix
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.dates as md
from pyfinance.ols import PandasRollingOLS

#import seaborn as sns
#sns.set(style="darkgrid")

start = '2017-01-30'
end = '2017-03-11'
sensor = 'tilt_magta'
seg_len = 1.5



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

def moving_ave(array):

    data = pd.DataFrame({'x':array})
    
    ma = data.rolling(7).mean()
    ma = np.array(ma)
    
    return ma

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



dg = data.groupby('node_id')
                
def velocity(df):
    try:
        #################################### xy
        ### lws
        lws_xy = low_ess(df.xy, np.arange(len(df.xy)), fraction = 0.09)
        lws_xy = lws_xy[:,1]
        df['lws_xy'] = np.array(lws_xy)
        vel_lws_xy = PandasRollingOLS(y=df.lws_xy,x=df.td, window=7)
        vel_lws_xy = ([np.nan] * 6) + list(abs(vel_lws_xy.beta.values))
        df['vel_lws_xy']=np.array(vel_lws_xy)

        ### sma        
        df['sma_xy'] = moving_ave(df.xy)
        
        vel_sma_xy = PandasRollingOLS(y=df.sma_xy,x=df.td, window=7)
        vel_sma_xy = ([np.nan] * 6) + list(abs(vel_sma_xy.beta.values))
        df['vel_sma_xy']=np.array(vel_sma_xy)

        #################################### xz
        ### lws
        lws_xz = low_ess(df.xz, np.arange(len(df.xz)), fraction = 0.09)
        lws_xz = lws_xz[:,1]
        df['lws_xz'] = np.array(lws_xz)
        vel_lws_xz = PandasRollingOLS(y=df.lws_xz,x=df.td, window=7)
        vel_lws_xz = ([np.nan] * 6) + list(abs(vel_lws_xz.beta.values))
        df['vel_lws_xz']=np.array(vel_lws_xz)

        ### sma        
        df['sma_xz'] = moving_ave(df.xz)
        
        vel_sma_xz = PandasRollingOLS(y=df.sma_xz,x=df.td, window=7)
        vel_sma_xz = ([np.nan] * 6) + list(abs(vel_sma_xz.beta.values))
        df['vel_sma_xz']=np.array(vel_sma_xz)
        
    except:
        print ('error found')
        pass
    return df


dg = dg.apply(velocity)

n = 14

node = dg[dg.node_id == n]

node.sort_values('ts', axis=0, ascending=True, inplace=True, na_position='last')

#fig,axs = plt.subplots(2,1, sharex = True, sharey = False)
#fig.subplots_adjust(hspace=0)
#
#axs[0].plot(node.ts.values,node.vel_sma_xy, color=dyna_colors[2],label = 'sma_xy')
#axs[0].plot(node.ts.values,node.vel_lws_xy, color=dyna_colors[0], label = 'lws_xy')
#axs[0].legend(loc = 'upper right')
#axs[0].axvline(x='2017-05-01',  color = 'red',linewidth=2)
##axs[0].axvspan('2017-05-01','2017-09-01', facecolor='yellow', alpha=0.5)
#axs[0].set_title('MAGTA-{} (velocity)'.format(n), fontsize = 20)
#
#axs[1].plot(node.ts.values,node.vel_sma_xz, color=dyna_colors[2],label = 'sma_xz')
#axs[1].plot(node.ts.values,node.vel_lws_xz, color=dyna_colors[0],label = 'lws_xz')
#axs[1].legend(loc = 'upper right')
#axs[1].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#axs[1].axvline(x='2017-05-01',  color = 'red',linewidth=2)
##axs[1].axvspan('2017-05-01','2017-09-01', facecolor='yellow', alpha=0.5)

fig2, axs2 = plt.subplots(2, sharex = True, sharey = False)
fig2.subplots_adjust(hspace=0)

axs2[0].plot(node.ts.values, node.xy, color='grey',alpha=0.85,label='raw_xz')
#axs2[0].plot(node.ts.values,node.sma_xy, color=dyna_colors[2],label = 'sma_xy')
axs2[0].plot(node.ts.values,node.lws_xy, color=dyna_colors[0], label = 'lws_xy')
axs2[0].legend(loc = 'upper right')
#axs2[0].axvline(x='2017-05-01',  color = 'red',linewidth=2)
#axs[0].axvspan('2017-05-01','2017-09-01', facecolor='yellow', alpha=0.5)
axs2[0].set_title('MAGTA-{} (fitting)'.format(n), fontsize = 20)

axs2[1].plot(node.ts.values, node.xz, color='grey',alpha=0.85,label='raw_xz')
#axs2[1].plot(node.ts.values, node.sma_xz, color=dyna_colors[2],label='sma_xz')
axs2[1].plot(node.ts.values, node.lws_xz, color=dyna_colors[0],label='lws_xz')
axs2[1].legend(loc = 'upper right')
axs2[1].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#axs2[1].axvline(x='2017-05-01',  color = 'red',linewidth=2)
#axs2[1].axvspan('2017-05-01','2017-09-01', facecolor='yellow', alpha=0.5)


#color = sns.color_palette("BrBG", 7)
#axs2.plot(node.ts.values, node.xy, lw = 3,color = '#B39DDB' ,label='raw data')
#axs2.plot(node.ts.values, node.sma_xy, lw = 3,color = color[1],label='SMA')
#axs2.plot(node.ts.values, node.lws_xy, lw = 3,color = color[6],label='LOWESS')
#axs2.legend(loc = 'lower right', fontsize=20)
##axs2.set_title('MAGTA-{} (fitting)'.format(n), fontsize = 20)
#axs2.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#plt.rcParams['xtick.labelsize']=20
#plt.rcParams['ytick.labelsize']=20
#plt.ylabel('Position', fontsize = 25)
#plt.xlabel('Timestamp', fontsize = 25)
##axs2[0].axvline(x='2017-05-01',  color = 'red',linewidth=2)
##axs2[0].axvspan('2017-05-01','2017-09-01', facecolor='yellow', alpha=0.5)