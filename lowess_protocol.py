# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:24:18 2019
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
        xz = df.xz
    else:
        xz = df.xy
    td = df.td
    ts = df.ts
    
    m = window ###embedding dimension
    
    lws = []
    last_val=[]
    t_d = []
    t_s = []
    for i in  range(len(xz) - m):
        x = xz[i:i+m]
#        x.append(array[i:i+m])
        
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

############ fitting with LOWESS

#def rolling_lws(val, timestamp, df_td):
#    lws_val = []
#    last_val = []
#    ts = []
#    td = []
#    for i in range(len(val)):
#        fit_a = low_ess(np.array(val[i]), np.arange(len(val[i])), fraction = 1.1)
#        fit_a = fit_a[:,1]
#        
#        lws_val.append(fit_a)
#        end=len(lws_val[i]) - 1
#        last_val.append(fit_a[end])
#        
#    for j in range(len(timestamp)):
#        end = len(timestamp[j])-1
#        ts.append(timestamp[j][end])
#    
#    for k in range(len(df_td)):
#        end = len(df_td[k])-1
#        td.append(df_td[k][end])
#    return lws_val, last_val, ts, td


def velocity(df):
    
    vel_ = PandasRollingOLS(y=df.lws,x=df.td, window=7)
    vel_ = ([np.nan] * 6) + list(abs(vel_.beta.values))

    return vel_

def acceleration(vel_,td):
    start_index = len(td) - len(vel_)
    
    accel = PandasRollingOLS(y = pd.Series(vel_), 
                             x = pd.Series(td[start_index:]), window = 7)
    
    accel = accel.beta.values
    
    return accel

def spearman(lowess_):
    coef=[]
    pval=[]
    for i in lowess_:
        c,p = spearmanr(np.arange(0,len(i),1), i)
        coef.append(c)
        pval.append(p)
    coef = np.array(coef)
    pval = np.array(pval)
    return coef,pval


sensors = ['tilt_tuetb','tilt_magta','tilt_loota','tilt_lootb', 'tilt_dadtb']

start = '2016-01-01'
end = '2018-12-01'
sensor = sensors[0]
seg_len = 1.5

data = data_query(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(data, seg_len)

data['td'] = data.ts.values - data.ts.values[0]

data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1,'D'))

thresholds = np.arange(0.001,0.10, 0.001)

conf_act = []
conf_pred = []
for t in thresholds:
        
    act = []
    pred = []
    for i in np.arange(8,16,1):
        node = data[data.node_id == i]
    
    ######################################################################################## xz
        lws_xz, lst_val_xz, td, ts = rolling_lws(node,slope = 'xz', window=17)
        
        
        xz_df = pd.DataFrame({'lws':lst_val_xz, 'td':td})
        
        
        vel_xz = velocity(xz_df)
        accel_xz = acceleration(np.array(vel_xz),td)
        accel_xz = accel_xz.reshape(len(accel_xz),)
        c_xz,p_xz = spearman(lws_xz)
    #######################################################################################
        pred_xz = [1 if i >= t else 0 for i in vel_xz]
    ####################################################################################### xy
        lws_xy, lst_val_xy, td, ts = rolling_lws(node,slope = 'xy', window=17)
        
        xy_df = pd.DataFrame({'lws':lst_val_xy, 'td':td})
        
        
        vel_xy = velocity(xy_df)
        accel_xy = acceleration(np.array(vel_xy),td)
        accel_xy = accel_xz.reshape(len(accel_xy),)
        c_xy,p_xy = spearman(lws_xy)
    #######################################################################################
        pred_xy = [1 if i >= t else 0 for i in vel_xy]
    #######################################################################################
        start_index = len(c_xz) - len(accel_xz)
    #######################################################################################
        df = pd.DataFrame({'c_xz':c_xz[start_index:], 'c_xy':c_xy[start_index:],'accel_xz':accel_xz, 
                           'accel_xy':accel_xy, 'pred_xz':pred_xz[start_index:], 'pred_xy':pred_xy[start_index:]})
        
        df.loc[(df.c_xz >= 0.6) & (df.accel_xz > 0.015),
               'bin_xz'] = 1  
               
        df.loc[(df.c_xy >= 0.6) & (df.accel_xy > 0.015),
               'bin_xy'] = 1  
    
        df.loc[(df.bin_xz == 1.0) | (df.bin_xy == 1.0),
               'actual'] = 1  
    
        df.loc[(df.pred_xz == 1.0) | (df.pred_xy == 1.0),
               'predicted'] = 1  
    
        df = df.fillna(0)
        
        act.extend(df.actual)
        pred.extend(df.predicted)
            
    conf_act.append(act)
    conf_pred.append(pred)
    
conf_act = np.array(conf_act)
conf_pred = np.array(conf_pred)

tpr = []
fpr = []
fnr = []
tp = []
tn = []
fp = []
fn = []
for i in range(len(conf_pred)):
    
    TN, FP, FN, TP = confusion_matrix(conf_act[i], conf_pred[i]).ravel()
    
    
    TPR = (TP / (TP + FN))
    FPR = (FP / (FP + TN))
    FNR = (FN / (FN + TP))

    tpr.append(TPR)
    fpr.append(FPR)
    fnr.append(FNR)
    tp.append(TP)
    tn.append(TN)
    fp.append(FP)
    fn.append(FN)
    
tp = np.array(tp)
fp = np.array(fp)
tn = np.array(tn)
fn = np.array(fn)
    
conf_df = pd.DataFrame({'thresh':thresholds,'tpr':tpr, 'fpr':fpr, 'fnr':fnr})
final = pd.concat([conf_df, pd.DataFrame({'tpr':1, 'fpr':1, 'thresh':1, 'fnr':1},index=[0])])
final = pd.concat([final, pd.DataFrame({'tpr':0, 'fpr':0, 'thresh':0, 'fnr':0},index=[0])])


final = final.sort_values(['fpr'])
final = final[['tpr', 'fpr', 'fnr', 'thresh']]

final.to_csv('{}-ROC.csv'.format(sensor))


plt.plot(final.fpr,final.tpr, color = 'blue', marker='o')
plt.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
plt.title('ROC Plot ({})'.format(sensor), fontsize = 30)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.ylabel('TPR', fontsize = 25)
plt.xlabel('FPR', fontsize = 25)


############################################################################### MCC score
MCC_n = ((tp)*(tn)) - ((fp)*(fn))
MCC_d = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)

MCC = MCC_n / MCC_d
df_mcc = pd.DataFrame({'thresh':thresholds, 'MCC':MCC})
df_mcc.to_csv('{}-MCC.csv'.format(sensor))

fig, ax = plt.subplots(1, sharex = True, sharey = False)
fig.subplots_adjust(hspace = 0, wspace=.001)
ax.plot(df_mcc.thresh, df_mcc.MCC, color ='#EF7215', marker = 'o')
ax.set_title('MCC vs thresh - {}'.format(sensor), fontsize = 35)
ax.set_ylabel('MCC Score', fontsize = 25)
ax.set_xlabel(r'threshold [$m/day$]', fontsize = 25)



############################################################################### F1 score
precision = tp / (tp + fp)
recall = TPR

f1 = 2 * ((precision * recall) / (precision + recall))
df_f1 = pd.DataFrame({'thresh':thresholds, 'f1':f1})
df_f1.to_csv('{}.csv'.format(sensor))

fig3, ax3 = plt.subplots(1, sharex = True, sharey = False)
fig3.subplots_adjust(hspace = 0, wspace=.001)
ax3.plot(thresholds, f1, color ='#0b6623', marker = 'o')
ax3.set_title('{}-F1.csv'.format(sensor), fontsize = 35)
ax3.set_ylabel('F1 Score', fontsize = 25)
ax3.set_xlabel(r'threshold [$m/day$]', fontsize = 25)
###############################################################################
