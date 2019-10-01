# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:14:09 2019
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")

sns.set(font_scale = 2)



mcc_magta = pd.read_csv('tilt_magta-MCC.csv')
mcc_magta = mcc_magta.fillna(0)

roc_magta = pd.read_csv('tilt_magta-ROC.csv')

#f1_magta = pd.read_csv('f1-tilt_magta')
#f1_magta = f1_magta.fillna(0)

fig, ax = plt.subplots(1, sharex = True, sharey = False)
fig.subplots_adjust(hspace = 0, wspace=.001)
ax.plot(mcc_magta.thresh, mcc_magta.MCC, color ='#0b6623', marker = 'o', label = 'MCC_magta')
ax.legend(loc = 'upper right')
ax.set_xlabel(r'threshold [$m/day$]', fontsize = 25)
ax.set_ylabel('MCC score', fontsize = 25)
ax.set_title('MCC vs Threshold', fontsize = 35)


fig2, ax2 = plt.subplots(1, sharex = True, sharey = False)
fig2.subplots_adjust(hspace = 0, wspace=.001)
ax2.plot(roc_magta.fpr, roc_magta.tpr, color ='#0b6623', marker = 'o', label = 'ROC-AUC_magta')
ax2.plot([0, 1], [0, 1],  linewidth = 2, linestyle = '-.', color = 'black')
ax2.legend(loc = 'lower right')
ax2.set_xlabel('FPR', fontsize = 25)
ax2.set_ylabel('TPR', fontsize = 25)
ax2.set_title('ROC-AUC', fontsize = 35)


#fig3, ax3 = plt.subplots(1, sharex = True, sharey = False)
#fig3.subplots_adjust(hspace = 0, wspace=.001)
#ax3.plot(f1_magta.thresh, f1_magta.f1, color ='#0b6623', marker = '+', label = 'f1_magta')
#ax3.legend(loc = 'upper right')
#ax3.set_xlabel(r'threshold [$m/day$]', fontsize = 25)
#ax3.set_ylabel('F1 score', fontsize = 25)
#ax3.set_title('F1 score vs Threshold', fontsize = 35)


roc_magta['dist'] = np.sqrt((roc_magta['fpr'] - 0)**2 + (roc_magta['tpr'] - 1)**2)
min_roc_magta = roc_magta[roc_magta.dist == min(roc_magta.dist)]
#max_f1_magta = f1_magta[f1_magta.f1 == max(f1_magta.f1)]
max_mcc_magta = mcc_magta[mcc_magta.MCC == max(mcc_magta.MCC)]



roc_magta['dist_fnr'] = np.sqrt((roc_magta['fpr'] - 0)**2 + (roc_magta['fnr'] - 0)**2)

roc_magta['LRp'] = roc_magta.tpr/(100 - (1 - roc_magta.fpr))
roc_magta['LRn'] = (100 - roc_magta.tpr)/(1 - roc_magta.fpr)
roc_magta['J'] = roc_magta.tpr - roc_magta.fpr

fig4, ax4 = plt.subplots(1, sharex = True, sharey = False)
fig4.subplots_adjust(hspace = 0, wspace=.001)
ax4.plot(roc_magta.thresh, roc_magta.J, color ='#0b6623', marker = 'o', label = 'f1_magta')
#ax4.legend(loc = 'upper right')
#ax4.set_xlabel(r'threshold [$m/day$]', fontsize = 25)
#ax4.set_ylabel('FNR', fontsize = 25)
#ax4.set_title('FNR score vs Threshold', fontsize = 35)
