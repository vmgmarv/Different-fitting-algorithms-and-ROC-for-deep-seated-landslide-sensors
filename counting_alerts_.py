# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:54:55 2019
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mysql.connector as sql


db_connection = sql.connect(host='127.0.0.1', database='senslopedb', 
                            user='root', password='senslope')
def q_sensors():
    read = db_connection.cursor()
    query = "SELECT tsm_name FROM tsm_sensors"   
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names
    
    sen = np.array(d.tsm_name)

    return sen


tsm = []
tn = []
fp = []
fn = []
tp = []
positive_pred = []
negative_pred = []

positive_act = []
negative_act = []

emp = pd.DataFrame()
sensors = q_sensors()




for i in sensors:
    try:
        sensor = i
        lws = pd.read_csv('tilt_{}.csv'.format(sensor))
        lws.sort_values(['ts'], inplace = True)
        lws['ts'] = pd.to_datetime(lws['ts'], errors='coerce')
        lws['ts']=lws['ts'].dt.round('30min')
        cur = pd.read_csv('percent_movement.csv')
        cur['ts'] = pd.to_datetime(cur['ts'],errors='coerce')
        
        
        df1 = cur[cur.tsm_name == sensor]
        
        new = df1.merge(lws.drop_duplicates(), on='ts', how='left', indicator=True)
        
        new = new[new._merge == 'both']
        
        new.drop(['lws_xy', 'lws_xz', 'node_id_x', 'node_id_y',
                  'vel_xz', 'vel_xy', 'td'], axis=1, inplace=True)
        
        
        new['actual'] = np.where(new['na_status'] == 1, 1,0)
        
        new['metrics'] = np.where(new['prediction'] == new['na_status'], 1, 0)
        
        
        act_pos = (new.actual == 1).sum()
        act_neg = (new.actual == 0).sum()
        
        
        pred_pos = (new.prediction == 1).sum()
        pred_neg = (new.prediction == 0).sum()

        TN, FP, FN, TP = confusion_matrix(new.actual, new.prediction).ravel()
        
        tn.append(TN)
        fp.append(FP)
        fn.append(FN)
        tp.append(TP)

        positive_pred.append(pred_pos)
        negative_pred.append(pred_neg)
        
        positive_act.append(act_pos)
        negative_act.append(act_neg)
        
        tsm.append(i)
        


    except:
        print('no alerts', i)
        pass
df = pd.DataFrame({'tsm':tsm, 'tn':tn,'fp':fp,'fn':fn,'tp':tp,'pred_p':positive_pred,
                   'pred_n':negative_pred, 'act_p':positive_act, 'act_n':negative_act})

ff = pd.DataFrame({'tsm':'humb', 'tn':[0],'fp':[0],'fn':[0],'tp':[0],'pred_p':[0],
                   'pred_n':[7], 'act_p':[0], 'act_n':[7]})                     ##### has error on ROC

gg = pd.DataFrame({'tsm':'ltetb', 'tn':[0],'fp':[0],'fn':[0],'tp':[0],'pred_p':[0],
                   'pred_n':[12], 'act_p':[0], 'act_n':[12]})                   ##### has error on ROC

    
df = pd.concat([df, ff])
df = pd.concat([df, gg])

df.to_csv('alerts_comparison.csv')
