# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:55:35 2019
"""
import pandas as pd
import numpy as np
import mysql.connector as sql
import filterdata as filt
import matplotlib.pyplot as plt
from pyfinance.ols import PandasRollingOLS
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.dates as md
from datetime import datetime

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
        d = d[d.type_num == nm]
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
        last_val.append(fit_a[-1])
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

def velocity(df):
   
    vel_xz = PandasRollingOLS(y=df.lws_xz,x=df.td, window=7)
    df['vel_xz'] = ([np.nan] * 6) + list(abs(vel_xz.beta.values))
    
    vel_xy = PandasRollingOLS(y=df.lws_xy,x=df.td, window=7)
    df['vel_xy'] = ([np.nan] * 6) + list(abs(vel_xy.beta.values))
    
    return df


def percent_movement(tsm_name,n):
    cur = pd.read_csv('percent_movement.csv')
    
    sen = cur[cur.tsm_name == tsm_name]
    sen['ts'] = sen['ts'].astype('datetime64[D]')
    
    ts_unq = np.unique(sen.ts)
    nodes = np.unique(sen.node_id)
    
    try:
        un_ts = np.array(ts_unq[n]).astype('datetime64[D]')
        start_ts = un_ts - np.timedelta64(5, 'D')
        end_ts = start_ts + np.timedelta64(15, 'D')

    except:
        print ('out of bounds')
        pass
    
    return start_ts, end_ts, nodes, sen.ts.iloc[n]
###############################################################################

def pre_proc(tsm_name, n):
    sensor = 'tilt_' + tsm_name

    start, end, nodes, ts = percent_movement(tsm_name, n)
    
    df = data_query(start, end, sensor)            
    filtered = filt.apply_filters(df)
    data = accel_to_lin_xz_xy(filtered, seg_len=1.5)
    
    data['td'] = data.ts.values - data.ts.values[0]
    data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1, 'D'))
    
    data = data[data['node_id'].isin(nodes)]
    data['xz'] = data.groupby('node_id')['xz'].transform(lambda v: v.ffill())
    data['xy'] = data.groupby('node_id')['xy'].transform(lambda v: v.ffill())

#    data['xy'].fillna(method='ffill', inplace = True)
#    data['xz'].fillna(method='ffill', inplace = True)
    
    return data, nodes, ts

    
tsm_name = 'humb'

df, node_id, ts = pre_proc(tsm_name, n = 2)


final = pd.DataFrame()

for i in node_id:
    node = df[df.node_id == i]
    node['ma_xz'] = moving_ave(node.xz)
    node['ma_xy'] = moving_ave(node.xy)
    lws_xz, last_val_xz, t_d, t_s = rolling_lws(node, slope = 'xz', window = 17)
    lws_xy, last_val_xy, t_d, t_s = rolling_lws(node, slope = 'xy', window = 17)
    
    daf = pd.DataFrame({'ts':t_s,'ma_xz':node.ma_xz.iloc[17:],'ma_xy':node.ma_xy.iloc[17:],
                        'td':t_d, 'lws_xz':last_val_xz, 'lws_xy':last_val_xy, 'node_id':i})
    vel_ = velocity(daf)
    
    daf['prediction'] = np.where((daf['vel_xz'] >= 0.032)|((daf['vel_xy']) >= 0.032), 1, 0)
    final = pd.concat([final, daf])

fig, (ax1,ax3) = plt.subplots(2,1, sharey = True)
fig.subplots_adjust(hspace = 0.01)

ax1.plot(final.ts, final.lws_xy, color = 'blue', label = 'LOWESS_xy')
ax1.plot(final.ts, final.ma_xy, color = 'orange', label = 'SMA_xy')
ax1.plot(df.ts, df.xy, color = 'grey', label = 'raw_xy')
ax1.set_ylabel('position')
ax2 = ax1.twinx()
ax2.plot(final.ts, final.prediction, '--ro',alpha = 0.5)
ax2.set_ylabel('prediction [0 or 1]', color = 'red')
#ax.plot(final.ts, final.)
ax1.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
ax1.legend(fontsize = 14)
ax1.axvline(ts, color = 'black')
ax1.xaxis.set_visible(False)


ax3.plot(final.ts, final.lws_xz, color = 'blue', label = 'LOWESS_xz')
ax3.plot(final.ts, final.ma_xz, color = 'orange', label = 'SMA_xz')
ax3.plot(df.ts, df.xz, color = 'grey', label = 'raw_xz')
ax3.set_ylabel('position')
ax4 = ax3.twinx()
ax4.plot(final.ts, final.prediction, '--ro',alpha = 0.5)
ax4.set_ylabel('prediction [0 or 1]', color = 'red')
#ax.plot(final.ts, final.)
ax3.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
ax3.axvline(ts, color = 'black')