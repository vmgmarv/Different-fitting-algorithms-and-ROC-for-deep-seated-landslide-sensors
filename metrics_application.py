# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:00:09 2019
"""

import matplotlib.pyplot as plt
import filterdata as filt
import pandas as pd
import numpy as np
import mysql.connector as sql
import fitting
from pyfinance.ols import PandasRollingOLS
import performance_metrics as metric
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
import math

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
    


start = '2017-01-01'
end = '2017-06-30'
sensor = 'tilt_magta'
seg_len = 1.5

node_num = 16 ##############################################################select node
thresh = 0.032 ############################################################### threshold

nodes = np.arange(10,18,1)

data = query_data(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(data, seg_len)

data['td'] = data.ts.values - data.ts.values[0]

data['td'] = data['td'].apply(lambda x:x / np.timedelta64(1,'D'))

lfa = []
sma = []
td = []

for i in nodes:
    node = data[data.node_id == i]
    node.sort_values('ts', axis=0, ascending=True,inplace=True, na_position='last')
    
    xz = np.array(node.xz)
    timestamp=np.array(node.ts.values)
    df_td = np.array(node.td.values)
    
    rol_xz = fitting.rolling_window(xz, window=17)
    timestamp = fitting.rolling_window(timestamp,window=17)
    df_td = fitting.rolling_window(df_td,window=17)
    lws_xz, lst_val, ts, n_td = fitting.rolling_lws(rol_xz, timestamp, df_td)
    lst_val = np.array(lst_val)
    td.append(n_td)
    lfa.append(lst_val)
    sma.append(fitting.moving_ave(xz))
    
sma = np.array(sma)
lfa = np.array(lfa)


############################################################################### ROC
nm = []
for k in range(len(lfa)):
    nm.append(len(lfa[k]))

threshold = np.arange(0.005,0.90,0.001)


velocity = []
acceleration = [] 
actual = []
predicted = []
thresh = []
col_pred = []
for c in range(len(lfa)):
    vel = PandasRollingOLS(y=pd.Series(lfa[c]),x=pd.Series(td[c]),window=7)
    vel = abs(vel.beta.values)
    
    vel= vel.reshape(len(vel))
    start_index = len(td[c]) - len(vel)
    
    t_d = td[c]
    accel = PandasRollingOLS(y=pd.Series(vel),x=pd.Series(t_d[start_index:]),window=7)
    accel = accel.beta.values
    
    start_index2 = len(vel) - len(accel)
    
    velocity.append(vel)
    acceleration.append(accel)
    
    cp = []
    for t in threshold:
        pred = metric.current_pred(vel[start_index:],t)
        
        cp.append(pred)
    predicted.append(cp)

acceleration = np.array(acceleration)

coef = []
pval = []
for i in range(len(lfa)):
    k = lfa[i]
    cc = []
    pp = []
    for j in range(len(k) - 7):
        kk = k[j:j+7]
        c,p = spearmanr(np.arange(0, len(kk), 1), kk)
        cc.append(c)
        pp.append(p)
    
    coef.append(cc)
    pval.append(pp)
    
coef = np.array(coef)

actual = []
col_act = []
for i in range(len(coef)):
    start_index = abs(len(coef[i]) - len(acceleration[i]))
    ac = acceleration[i]
    co = coef[i][start_index:]
    
    act_ = []
    for t in threshold:
        act1 = [1 if s >= 0.02 else 0 for s in ac]
        act2 = [1 if x >= 0.49 else 0 for x in co]
        
        act3 = np.array(act1) + np.array(act2)
        
        tot_act = [1 if i == 2 else 0 for i in act3]
        act_.append(tot_act)
    actual.append(act_)
    

    
    
    
    

######################################################################################
actual = np.array(actual)
predicted = np.array(predicted)

tp = []
fp = []
tn = []
fn = []
for m in range(len(threshold)):
    ttp = []
    ffp = []
    ttn = []
    ffn = []
    for n in range(len(lfa)):

        TN, FP, FN, TP = confusion_matrix(actual[n][m], predicted[n][m]).ravel()
        ttp.append(TP)
        ffp.append(FP)
        ttn.append(TN)
        ffn.append(FN)
 
    t_p = sum(ttp)
    f_p = sum(ffp)
    t_n = sum(ttn)
    f_n = sum(ffn)
    
    tp.append(t_p)
    fp.append(f_p)
    tn.append(t_n)
    fn.append(f_n)
    
tp = np.array(tp)
fp = np.array(fp)
tn = np.array(tn)
fn = np.array(fn)

TPR = (tp / (tp + fn))
FPR = (fp / (fp + tn))
FNR = (fn / (fn + tp))

#plt.plot(FPR,TPR, color = 'blue', marker='o')
#plt.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
df_roc = pd.DataFrame({'tpr':TPR,'fpr':FPR,'fnr':FNR,'thresh':threshold})
adnal1 = pd.DataFrame({'tpr':1, 'fpr':1, 'fnr':1,'thresh':1},index=[0])
adnal2 = pd.DataFrame({'tpr':0, 'fpr':0, 'fnr':0,'thresh':0},index=[0])


df_roc = pd.concat([df_roc, adnal1])
df_roc = pd.concat([df_roc, adnal2])
df_roc = df_roc.sort_values(['fpr'])
df_roc.to_csv('ROC-{}'.format(sensor))
#final = final[['tpr', 'fpr', 'thresh']]

fig2, ax2 = plt.subplots(1,sharex = True, sharey = False)
fig2.subplots_adjust(hspace = 0, wspace =.001)
ax2.plot(df_roc.fpr,df_roc.tpr, marker = 'o', color = '#0080FF')
ax2.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
ax2.set_title('ROC - {}'.format(sensor), fontsize = 35)
ax2.set_ylabel('TPR', fontsize = 25)
ax2.set_xlabel('FPR', fontsize = 25)

################################################################################ MCC

MCC_n = ((tp)*(tn)) - ((fp)*(fn))
MCC_d = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)

MCC = MCC_n / MCC_d
df_mcc = pd.DataFrame({'thresh':threshold, 'MCC':MCC})
df_mcc.to_csv('MCC-{}'.format(sensor))

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
df_f1 = pd.DataFrame({'thresh':threshold, 'f1':f1})
df_f1.to_csv('f1-{}'.format(sensor))

fig3, ax3 = plt.subplots(1, sharex = True, sharey = False)
fig3.subplots_adjust(hspace = 0, wspace=.001)
ax3.plot(threshold, f1, color ='#0b6623', marker = 'o')
ax3.set_title('F1 - {}'.format(sensor), fontsize = 35)
ax3.set_ylabel('F1 Score', fontsize = 25)
ax3.set_xlabel(r'threshold [$m/day$]', fontsize = 25)
###############################################################################


