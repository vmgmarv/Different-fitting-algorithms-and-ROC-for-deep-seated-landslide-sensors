# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:26:35 2019
"""

import pandas as pd
import numpy as np
import filterdata as filt
from scipy.signal import savgol_filter
import mysql.connector as sql
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.cm as cm
from scipy import stats, polyval

sns.set_style("darkgrid")

db_connection = sql.connect(host='192.168.150.75', database='senslopedb', 
                            user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

start = '2017-01-01'
end = '2018-10-30'
sensor = 'tilt_marta'
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

def fitting(array):
    '''
    for savgol:
        window_length = 7
        polyorder = 4
    '''
    ############################### Moving Average
    ma = array.rolling(12).mean()
    ma = np.array(ma)
    
    ############################### SG filter
    sg = savgol_filter(array, window_length=13, polyorder=5)
    return ma,sg


def regression(x):
    
    regression = []
    slope = []
    
    for i in x:
        x_new = np.arange(0,len(i))
        (a_s, b_s, r, tt, stderr) = stats.linregress(x_new, i)
        reg = polyval([a_s,b_s],x_new)
        reg = reg.tolist()  
        regression.append(reg)
        slope.append(a_s)
    
    reg = np.array(regression)
    slope = np.array(slope)
    
    return reg, slope


def get_features(array):
    
    x = []
    y = []
    z = []
    
    m = 144 ##### 3 day data (144), 2 day data (96)
    t = 145 ##### for extension
    
    for i in range(len(array) - t):
        
        x.append(array[i:i+m])############################### Gets 3 day data
        y.append(array[i+m]) 
        z.append(array[i+t] - array[i+m]) ################### Calculate displacement of the next data after 3 days
        
    x = np.array(x)
    y = np.array(y)
    z = abs(np.array(z))
    
    return x, y, z


def actual_event(array, thresh):
    '''
    array = z
    positive event if z >= threshold
    '''
    
    actual = []
    
    for i in array:
        actual.append(1 if i >= thresh else 0)
    
    actual = np.array(actual)
    
    return actual


def current_prediction(x, actual, thresh):
    
    '''
    Calculates displacement within 3 day data
    prediction is 1 if exceeded threshold else 0
    '''
    
    predicted = []
    for i in range(len(x)):
        x_pred = x[i][143] - x[i][0] #################### 3rd day [144] - 1day [0]
        x_pred = abs(x_pred)
        
        if x_pred >= thresh:
            ins = 1 ######################### Predicted True
        
        else:
            ins = 0 ######################### Predicted False
        
        predicted.append(ins)
    predicted = np.array(predicted)
    
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR



def reg_prediction(slope, thresh, actual):
    
    '''
    Returns calculated FPR, TPR
    '''
    predicted = []
    
    for i in slope:
        predicted.append(1 if i >= thresh else 0) ############### Check if slope exceeded threshold
    predicted = np.array(predicted)
        
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    TPR = (tp / (tp + fn))
    FPR = (fp / (fp + tn))
    
    return TPR, FPR


data = data(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(filtered, seg_len)

n = data.node_id.max() + 1 
cols = np.arange(1,n)


cur_tpr = []
cur_fpr = []
reg_tpr = []
reg_fpr = []
n_id = []
n_id2 = []
thresh = []
thresh2 = []


threshold = np.arange(0.00001,0.06,0.00001)

for j in threshold:
        t = j
        for i in cols:
            node = data[data.node_id == i]
            if node.empty:
                i += 1
                
            else:
                xz = node.xz
                ma,sg = fitting(xz)
                try:
                    x,y,z = get_features(ma)
                    actual = actual_event(z, t)################################ True event
                    current_TPR, current_FPR = current_prediction(x,actual,t)
                    
#                    regr, slope = regression(x)
#                    regression_tpr, regression_fpr = reg_prediction(slope, t, actual)

                    cur_tpr.append(current_TPR)
                    cur_fpr.append(current_FPR)
                    
#                    reg_tpr.append(regression_tpr)
#                    reg_fpr.append(regression_fpr)
                    
                    n_id.append(node.node_id.max())
                    thresh.append(t)
                except:
                    pass
                i += 1
        
        for l in cols:
            node2 = data[data.node_id == l]
            
            if node.empty:
                l += 1
                
            else:
                xz = node2.xz
                ma,sg = fitting(xz)
                
                try:
                    x,y,z = get_features(ma)
                    actual = actual_event(z,t)
                    
                    regr, slope = regression(x)
                    regression_tpr, regression_fpr = reg_prediction(slope, t, actual)
                    
                    reg_tpr.append(regression_tpr)
                    reg_fpr.append(regression_fpr)
                    
                    n_id2.append(node2.node_id.max())
                    thresh2.append(t)
                    
                except:
                    pass
                l += 1
                        
                
###################################################################################### Current
df_current = pd.DataFrame({'id':n_id, 'tpr':cur_tpr, 'fpr':cur_fpr, 'thresh':thresh2})

df_current = df_current[['id', 'tpr', 'fpr', 'thresh']]
df_current = df_current.fillna(0)

n2 = df_current.id.max() + 1
cols2 = np.arange(1,n2)

df_current = df_current.sort_values('thresh')
print ('Best thresholds: \n',df_current.nlargest(4, 'tpr'))

###################################################################################### Regression

df_reg = pd.DataFrame({'id':n_id2, 'tpr':reg_tpr, 'fpr':reg_fpr, 'thresh':thresh})
df_reg = df_reg[['id', 'tpr', 'fpr', 'thresh']]

###################################################################################### Plotting

colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
for i in cols2:
    n_node = df_current[df_current.id == i]
    adnal1 = pd.DataFrame({'id':i, 'tpr':1, 'fpr':1, 'thresh':1},index=[0])
    adnal2 = pd.DataFrame({'id':i, 'tpr':0, 'fpr':0, 'thresh':1},index=[0])
    final = pd.concat([n_node, adnal1])
    final = pd.concat([final, adnal2])
    final = final.sort_values(['fpr'])

    if n_node.empty:
        i += 1
    
    else:
        plt.plot(final.fpr,final.tpr, linewidth = 2, color = next(colors), marker = 'o', alpha = 0.9, label = 'Node {}'.format(i))
        plt.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
        plt.legend()
plt.title('ROC Plot (Current State)', fontsize = 30)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.ylabel('TPR', fontsize = 25)
plt.xlabel('FPR', fontsize = 25)



df_current['rank'] = df_current['tpr'].rank(ascending=0)
df_current =df_current.drop_duplicates(subset = 'thresh')
rank = df_current.loc[(df_current['rank'] <= 20)]
