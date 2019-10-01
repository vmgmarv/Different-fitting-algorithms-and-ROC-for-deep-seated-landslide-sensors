# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:22:47 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md


sensor = 'tueta'
lws = pd.read_csv('tilt_{}.csv'.format(sensor))
lws.sort_values(['ts'], inplace = True)
lws['ts'] = pd.to_datetime(lws['ts'], errors='coerce')
lws['ts']=lws['ts'].dt.round('1440min')
cur = pd.read_csv('percent_movement.csv')
cur['ts'] = pd.to_datetime(cur['ts'],errors='coerce')
cur['ts'] = cur['ts'].dt.round('1440min')

df1 = cur[cur.tsm_name == sensor]

new = df1.merge(lws.drop_duplicates(), on='ts', how='left', indicator=True)

new = new[new._merge == 'both']

new.drop(['lws_xy', 'lws_xz', 'node_id_x', 'node_id_y',
          'vel_xz', 'vel_xy', 'td'], axis=1, inplace=True)


new['actual'] = np.where(new['na_status'] == 1, 1,0)

new['metrics'] = np.where(new['prediction'] == new['na_status'], 1, 0)


fig, ax = plt.subplots()

p1 = ax.plot(new.ts, new.actual,'--ro', label = 'SMA')


p2 = ax.plot(new.ts, new.prediction,'--bo', label = 'LOWESS')

ax.set_title('{}'.format(sensor))
ax.set_ylabel('Count', fontsize = 15)

ax.legend(fontsize = 15, loc = 'upper right')
ax.autoscale_view()

plt.show()

ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))

'''
Iba na itong code sa baba.. magtitingin lang ng nodes na nag triggers
'''

df = pd.read_csv('percent_movement.csv')
df['ts'] = pd.to_datetime(df['ts']).astype('datetime64[D]')


unique_sen = np.unique(df.tsm_name)

final_orig = pd.DataFrame()
for i in unique_sen:
    sencur = df[df.tsm_name == i]
    
    ts_unique = np.unique(sencur.ts).astype('datetime64[D]')
    
    ts_unique = pd.Series(ts_unique)
    
    day = pd.Timedelta('1d')
    oo = pd.DataFrame({'ts':ts_unique, 'tsm_name':i})
    
    
    oo['bin_or'] = np.where((oo['ts'].shift(1) - oo['ts']).abs() != day, 1, 0)
    
    o_t = oo[oo.bin_or == 1]
    
    final_orig = pd.concat([final_orig, o_t])
    
    print (i)


original = final_orig.groupby(['tsm_name','bin_or']).size().reset_index(name='counts_or')
original.drop(['bin_or'], axis = 1, inplace = True)
'''
Lowess part, magulo na siya ayusin mo pls

'''


df_lws = pd.read_csv('using_lowess.csv')
df_lws['ts'] = pd.to_datetime(df_lws['ts']).astype('datetime64[D]')

unique_sen_lws = np.unique(df_lws.tsm_name)

final_lws = pd.DataFrame()
for j in unique_sen_lws:
    senlws = df_lws[df_lws.tsm_name == j]
    
    ts_unique_lws = np.unique(senlws.ts).astype('datetime64[D]')
    
    ts_unique_lws = pd.Series(ts_unique_lws)

    day = pd.Timedelta('1d')
    pp = pd.DataFrame({'ts':ts_unique, 'tsm_name':j})

    pp['bin_lws'] = np.where((pp['ts'].shift(1) - pp['ts']).abs() != day, 1, 0)
    
    o_l = pp[pp.bin_lws == 1]
    
    final_lws = pd.concat([final_lws, o_l])
    
    print (j)


sp = ['nagsa', 'lipt', 'ltetb', 'humb', 'labt']


for k in sp:
    df_s = pd.read_csv('tilt_{}.csv'.format(k))
    df_s['ts'] = pd.to_datetime(df_s['ts']).astype('datetime64[D]')
    ts_unique_s = np.unique(df_s.ts).astype('datetime64[D]')
    
    ts_unique_s = pd.Series(ts_unique_s)
    ss = pd.DataFrame(pd.DataFrame({'ts':ts_unique_s, 'tsm_name':k}))
    ss['bin_lws'] = np.where((ss['ts'].shift(1) - ss['ts']).abs() != day, 1, 0)
    
    s_l = ss[ss.bin_lws == 1]
    
    final_lws = pd.concat([final_lws, s_l])
    
    print('special', k)



'''
for special cases
'''
sp2 = 'magta'
df_s2 = pd.read_csv('{}.csv'.format(sp2)) 

df_s2['ts'] = pd.to_datetime(df_s2['ts']).astype('datetime64[D]')
ts_unique_s2 = np.unique(df_s2.ts).astype('datetime64[D]')

ts_unique_s2 = pd.Series(ts_unique_s2)
ss2 = pd.DataFrame(pd.DataFrame({'ts':ts_unique_s2, 'tsm_name':sp2}))
ss2['bin_lws'] = np.where((ss2['ts'].shift(1) - ss2['ts']).abs() != day, 1, 0)

s_l2 = ss2[ss2.bin_lws == 1]
#
final_lws = pd.concat([final_lws, s_l2])

lowess = final_lws.groupby(['tsm_name','bin_lws']).size().reset_index(name='counts_lws')
lowess.drop(['bin_lws'], axis = 1, inplace = True)


result = lowess.merge(original, on='tsm_name', how='left')


N = len(result.tsm_name)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, result.counts_or, width, color = '#CBE86B', label = 'SMA')


p3 = ax.bar(ind+width, result.counts_lws, width, color = '#182945', label = 'LOWESS')


ax.set_title('No. of Events per site')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels((result.tsm_name), rotation=45)
ax.set_ylabel('Count', fontsize = 15)

ax.legend(fontsize = 15, loc = 'upper right')
ax.autoscale_view()

plt.show()
plt.grid()


fig2, ax2 = plt.subplots()
label = ['SMA','LOWESS']
smt = [sum(result.counts_or),sum(result.counts_lws)]

colors = ['#16526D','#F8991D']
ax2.pie(smt, labels = label, autopct='%1.2f%%', shadow=True, startangle=140, colors = colors)
ax2.axis('equal')
plt.show()


