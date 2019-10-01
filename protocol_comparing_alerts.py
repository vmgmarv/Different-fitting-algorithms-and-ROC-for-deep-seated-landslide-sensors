# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:58:00 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('alerts_comparison.csv')
#df = pd.DataFrame({'tsm':tsm, 'tn':tn,'fp':fp,'fn':fn,'tp':tp,'pred_p':positive_pred,
#                   'pred_n':negative_pred, 'act_p':positive_act, 'act_n':negative_act})

ff = pd.DataFrame({'tsm':'humb', 'tn':[0],'fp':[0],'fn':[0],'tp':[0],'pred_p':[0],
                   'pred_n':[7], 'act_p':[0], 'act_n':[7]})                     ##### has error on ROC

gg = pd.DataFrame({'tsm':'ltetb', 'tn':[0],'fp':[0],'fn':[0],'tp':[0],'pred_p':[0],
                   'pred_n':[12], 'act_p':[0], 'act_n':[12]})                   ##### has error on ROC

df = pd.concat([df, ff])
df = pd.concat([df, gg])

    

####################################################### Bar chart
N = len(df.tsm)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, df.act_p, width, color = '#CBE86B', label = 'Actual Positive')


p2 = ax.bar(ind, df.act_n, width, bottom = df.act_p, color = '#D1F1FC', label = 'Actual Negative')


p3 = ax.bar(ind+width, df.pred_p, width, color = '#182945', label = 'Predicted Positive')


p4 = ax.bar(ind+width, df.pred_n, width, bottom = df.pred_p, color = '#991B1E', label = 'Predicted Negative')

ax.set_title('No. of Positive and Negative per site')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels((df.tsm), rotation=45)
ax.set_ylabel('Count', fontsize = 15)

ax.legend(fontsize = 15, loc = 'upper right')
ax.autoscale_view()

plt.show()
plt.grid()
########################################################## pie chart

fig2, ax2 = plt.subplots()
prediction = 'LOWESS Positive','LOWESS Negative'
actual = 'SMA Positive','SMA Negative'
val_pred = [sum(df.pred_p), sum(df.pred_n)]
val_act = [sum(df.act_p), sum(df.act_n)]

colors = ['#16526D','#F8991D']
ax2.pie(val_act, labels = actual, autopct='%1.2f%%', shadow=True, startangle=140, colors = colors)
ax2.axis('equal')
plt.show()
####################################################################
