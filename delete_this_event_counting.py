# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:31:48 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md


result = pd.read_csv('result_event.csv')

result = result.loc[:, ~result.columns.str.contains('^Unnamed')]

result = result[result.tsm_name != 'umita']
result = result[result.tsm_name != 'plab']
result = result[result.tsm_name != 'gamb']
result = result[result.tsm_name != 'pngta']


N = len(result.tsm_name)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, result.bin_c_x, width, color = '#CBE86B', label = 'SMA')


p3 = ax.bar(ind+width, result.bin_c_y, width, color = '#182945', label = 'LOWESS')


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
smt = [sum(result.bin_c_x),sum(result.bin_c_y)]

colors = ['#16526D','#F8991D']
ax2.pie(smt, labels = label, autopct='%1.2f%%', shadow=True, startangle=140, colors = colors)
ax2.axis('equal')
plt.show()
