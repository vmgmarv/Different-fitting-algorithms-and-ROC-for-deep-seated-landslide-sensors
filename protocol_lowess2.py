# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:14:11 2019
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
import rolling
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale = 2)



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

def query_data(start, end, sensor):
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

def moving_ave(array):

    data = pd.DataFrame({'x':array})
    
    ma = data.rolling(7).mean()
    ma = np.array(ma)
    
    return ma


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


def velocity(df):
    
    vel_ = PandasRollingOLS(y=df.val,x=df.td, window=7)
    vel_ = ([np.nan] * 6) + list(abs(vel_.beta.values))

    return vel_



start = '2017-04-27'
end = '2017-05-8'
sensor = 'tilt_magta'
seg_len = 1.5

node_num = 16 ##############################################################select node
thresh = 0.032 ############################################################### threshold



data = query_data(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(data, seg_len)

data['td'] = data.ts.values - data.ts.values[0]

data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1,'D'))


node = data[data.node_id == node_num]
node.sort_values('ts', axis=0, ascending=True, inplace=True, na_position='last')

xz = np.array(node.xz)

xy = np.array(node.xy)
timestamp = np.array(node.ts.values)
df_td = np.array(node.td.values)


###############################################################################fitting
sma_xz = moving_ave(xz)


rol_xz1 = rolling_window(np.array(node.xz), window = 7)
timestamp1 = rolling_window(np.array(node.ts.values), window = 7)
df_td1 = rolling_window(np.array(node.td.values), window=7)
lws_xz1, lst_val1, ts1, td1 = rolling_lws(rol_xz1,timestamp1, df_td1)

rol_xz2 = rolling_window(xz, window = 17)
timestamp2 = rolling_window(timestamp, window=17)
df_td2 = rolling_window(df_td, window=17)
lws_xz2, lst_val2, ts2, td2 = rolling_lws(rol_xz2,timestamp2,df_td2)

rol_xz3 = rolling_window(xz, window = 29)
timestamp3 = rolling_window(timestamp, window=29)
df_td3 = rolling_window(df_td, window=29)
lws_xz3, lst_val3, ts3, td3 = rolling_lws(rol_xz3,timestamp3,df_td3)
############################################################################### no window for lowess
xz_lws = low_ess(xz, timestamp, fraction = 0.1)
df_lws = pd.DataFrame({'val':xz_lws[:,1], 'td':df_td})
vel = np.array(velocity(df_lws))
vel = vel*100


################################################################################## velocity
df1 = pd.DataFrame({'val':lst_val1, 'td':td1})
df2 = pd.DataFrame({'val':lst_val2, 'td':td2})
df3 = pd.DataFrame({'val':lst_val3, 'td':td3})
df_sma = pd.DataFrame({'val':sma_xz.reshape(len(sma_xz)), 'td':node.td})



vel1 = np.array(velocity(df1))
vel2 = np.array(velocity(df2))
vel3 = np.array(velocity(df3))
vel_sma = np.array(velocity(df_sma))
##################################################################################       
fig, ax = plt.subplots(1, sharex = True, sharey = False)
fig.subplots_adjust(hspace = 0, wspace=.001)

lns1 = ax.plot(node.ts.values, xz, color ='grey', label = 'raw', linewidth = 2)
lns2 = ax.plot(node.ts.values, sma_xz, color = 'red', label = 'SMA')
lns3 = ax.plot(ts1, lst_val1, color = '#EF7215', label = '3-hour window')
lns4 = ax.plot(ts2, lst_val2, color = '#CBE86B', label = '8-hour window')
lns5 = ax.plot(ts3, lst_val3, color = '#182945', label = '14-hour window')               

#lns3 = ax.plot(node.ts.values, xz_lws[:,1], color = '#EF7215', label = 'LOWESS')
#ax2 = ax.twinx()
#lns3 = ax2.plot(node.ts.values, vel_sma, '--r', label='Velocity', alpha = 0.65)
ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))


lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0, fontsize = 20)

ax.set_xlabel('Timestamp', fontsize = 25)
ax.set_ylabel('Position', fontsize = 25)


node['lws'] = np.append(np.repeat(np.nan, (len(node) - len(lst_val2))),lst_val2)
#ax2.set_ylabel(r'Velocity[$cm/day$]', fontsize = 25)

fig2, ax22 = plt.subplots(1,sharex = True, sharey = False)
#ax22.plot(node.ts.values, xz, color = 'grey', label = 'raw', linewidth = 2)
ll1 = ax22.plot(node.ts.values, vel_sma, color = 'red', label = 'SMA')
ll2 = ax22.plot(ts1, vel1,color='#EF7215', label = '3-hour window')
ll3 = ax22.plot(ts2, vel2,color='#CBE86B', label = '8-hour window')
ll4 = ax22.plot(ts3, vel3,color='#182945', label = '14-hour window')

lls = ll1+ll2+ll3+ll4
labs2 = [l.get_label() for l in lls]
ax22.axhline(y=0.032, label = 'threshold', color = 'black', linestyle = '--')
ax22.legend(lls, labs2, loc=0, fontsize = 20)

ax22.set_xlabel('Timestamp', fontsize = 25)
ax22.set_ylabel(r'Velocity [$m/day$]', fontsize = 25)
ax22.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))

#ax22.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#ax22.set_ylabel(r'Velocity[$cm/day$]', fontsize = 25)
#ax22.set_xlabel('Timestamp', fontsize = 25)
#ax22.legend(fontsize=20, loc = 'upper left')
#ax[0].plot(ts1, lst_val1,color = '#0080FF', label = '3-hr')
#ax[0].plot(ts2, lst_val2,color = '#EF7215', label = '9-hr')
#ax[0].plot(ts3, lst_val3,color = '#0b6623', label = '14-hr')
#ax[0].plot(node.ts.values, sma_xz, alpha = 0.85, color='red', label = 'sma')
#ax[0].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#ax[0].legend(loc = 'upper left')
#ax[0].set_title('{}-{}'.format(sensor,max(node.node_id), fontsize = 35, y=1.2))


##fig2, ax2 = plt.subplots(1,sharex = True, sharey = False)
#
#ax[1].plot(node.ts.values, vel_sma, '--bo', color = 'grey', label = 'SMA')
#ax[1].plot(ts1, vel1, '--bo', color = '#0080FF', label = '3-hr')
#ax[1].plot(ts2, vel2, '--bo', color = '#EF7215', label = '9-hr')
#ax[1].plot(ts3, vel3, color = '#0b6623', label = '14-hr')
#ax[1].xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
##plt.legend(loc = 'upper right')
##ax2.set_title('Velocity magta-{}'.format(max(node.node_id), fontsize = 35, y=1.2))
#ax[1].axhline(y=thresh,  color = 'red')
#
#ax[1].legend(loc = 'upper left')
##ax3 = ax2.twinx()
##ax3.plot(node.ts.values, xz, alpha=0.85, color= 'black', label = 'raw')
##plt.legend(loc = 'upper right')
##ax2.axvline(x='2017-06-10 19:30', color = 'red')


################################################################################## time delay
#
#sec = np.timedelta64(ts1[3] - ts2[3],'s')
#sec = sec.astype('timedelta64[s]')
#sec = sec/np.timedelta64(1,'s')
#
#t0 = [] ##### vel for sma
#t1 = [] ##### vel for vel1
#t2 = [] #####
#
##start_index = len(vel1) - len(vel2)
#
##vel1 = np.column_stack((np.array(ts1),vel1))
##vel2 = np.column_stack((np.array(ts2),vel2))
#
#
#for i in np.arange(0,len(vel_sma)): ####################### SMA
#    if vel_sma[i] >= thresh:
#        t0.append(node.ts.values[i])
#
#
#for j in np.arange(0,len(vel1)): ########################## 3-hr
#    if vel1[j] >= thresh:
#        t1.append(ts1[j])
#
#
#for k in np.arange(0,len(vel2)): ########################## 3-hr
#    if vel2[k] >= thresh:
#        t2.append(ts2[k])
#        
#        
#delay_3hr = np.timedelta64(t1[0] - t0[0], 's').astype('timedelta64[s]')
#delay_3hr = delay_3hr/np.timedelta64(1,'s')
#print('SMA vs 3-hr window delay = ', delay_3hr)
#
#
#
#delay_9hr = np.timedelta64(t2[0] - t0[0], 's').astype('timedelta64[s]')
#delay_9hr = delay_9hr/np.timedelta64(1,'s')
#print('SMA vs 9-hr window delay = ', delay_9hr)

#plt.axvline('2017-03-13 14:00:00')
#start_index4 = len(xz) - (len(sma_xz))
#sma_xz = sma_xz.reshape(len(sma_xz))
#MA = pd.DataFrame({'y':xz, 'ma':sma_xz})
#MA = MA.dropna()
#RMSE_sma = math.sqrt(mean_squared_error(MA.y,MA.ma))
#
#RMSE = []
#data_span = []
#for i in np.arange(7,150):
#    rol_xz = rolling_window(xz, window=int(i))
#    timestamp = rolling_window(timestamp, window=int(i))
#    lws_xz,lst_val,ts = rolling_lws(rol_xz, timestamp)
#    
#    start_index = len(xz) - (len(lst_val))
#    rmse_ = math.sqrt(mean_squared_error(xz[start_index:],lst_val))
#    
#    data_span.append(i)
#    RMSE.append(rmse_)
#    print('{} done'.format(i))
#
#RMSE = np.array(RMSE)
#data_span = np.array(data_span)
#
#
#fig2, ax2 = plt.subplots(1, sharex = True, sharey = False)
#ax2.plot(data_span, RMSE)
#ax2.text(0.95, 0.01, 'RMSE_SMA = {0:.6f}'.format(RMSE_sma),
#        verticalalignment='bottom', horizontalalignment='right',
#        transform=ax2.transAxes,
#        color='green', fontsize=15)
#
#plt.xlabel('Data span', fontsize = 25)
#plt.ylabel('RMSE score', fontsize = 25)
#plt.title('RMSE magta {}'.format(max(node.node_id)))
#import win_mario