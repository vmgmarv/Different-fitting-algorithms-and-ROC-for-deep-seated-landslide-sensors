# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:14:24 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import fitting as fit
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV


x1 = np.linspace(-4, 4, 1000)
x2 = np.linspace(4, 8, 1000)
y1 = (8 * x1**5) - (20*(x1**4)) + (5 * (x1**3)) + (50 *(x1**2) - (20*x1)) -40
y2 = -2*x1**4 + 4*x1**3
#y2 = 10*np.sin(x1/(2*np.pi))

mean_noise = 20
mean_noise2 = 50
noise1 = np.random.normal(mean_noise, 200, len(y1))
noise2 = np.random.normal(mean_noise2, 550, len(y2))

y1_noise = y1 + noise1
y2_noise = -y1-10000 + noise2

x = np.concatenate((x1,x2))
y = np.concatenate((y1_noise,y2_noise))
############################################################################### fitting

ma = fit.moving_ave(y)
sg = fit.savitzky_golay(y)
lowess = fit.low_ess(y, x, fraction = 0.05)
ema = fit.ema(y)
ann = fit.ann(y)
dn_ann = (ann) * (max(y) - min(y)) + min(y)
start_index = len(x) - (len(dn_ann))

############################################################################### RMSE

### for MA
ma = ma.reshape(2000)
MA = pd.DataFrame({'y':(np.concatenate((y1,y2))), 'ma':ma})
MA = MA.dropna()
RMSE_ma = math.sqrt(mean_squared_error(MA.y,MA.ma))
print ('##### RMSE_ma = %.3f #####'%(RMSE_ma))
       
### for loess
RMSE_loess = math.sqrt(mean_squared_error((np.concatenate((y1,y2))), (lowess[:,1])))
print ('##### RMSE_loess = %.3f #####'%(RMSE_loess))

### for SG
RMSE_sg = math.sqrt(mean_squared_error((np.concatenate((y1,y2))),sg))
print ('##### RMSE_sg = %.3f #####'%(RMSE_sg))
       
### for EMA
EMA = np.array(ema.x)
RMSE_ema = math.sqrt(mean_squared_error((np.concatenate((y1,y2))),EMA))
print ('##### RMSE_ema = %.3f #####'%(RMSE_ema))
       
### for ANN
RMSE_ann = math.sqrt(mean_squared_error((np.concatenate((y1,y2)))[start_index:],dn_ann))
print ('##### RMSE_ann = %.3f #####'%(RMSE_ann))



textstr = '\n'.join((
    r'$\mathtt{MA} = %.3f$' % (RMSE_ma, ),
    r'$\mathtt{Loess} = %.3f$' % (RMSE_loess, ),
    r'$\mathtt{SG} = %.3f$' % (RMSE_sg, ),
    r'$\mathtt{EMA} = %.3f$' % (RMSE_ema, ),
    r'$\mathtt{ANN} = %.3f$' % (RMSE_ann, )))


############################################################################### Plotting

fig, (axes) = plt.subplots(6, 1, sharex = True)

axes[0].set_title(r'$5x^5 - 20x^4 + 5x^3 + 50x^2 - 20x - 40$',
                    fontsize = 30, y=1.2)
#axes[0].set_title(r'$\mathcal{10} \mathit{sin} (\frac{x}{2 \pi})$',
#                    fontsize = 30, y=1.2)
axes[0].plot(x,y, label='Noise')
axes[0].plot(x2,y2,color='red',label ='Actual')
axes[0].legend(loc = 'upper right')

axes[1].plot(x2,ma,color='yellow',label='Moving Ave')
#axes[1].legend(bbox_to_anchor=(1, 1.05), fancybox=True, shadow=True)
axes[1].legend(loc = 'upper right')

axes[2].plot(lowess[:, 0], lowess[:, 1],color='blue',label='Loess')
axes[2].legend(loc = 'upper right')
axes[2].text((x2[0]-1.25), (y2.max()-2), textstr, style='oblique', fontsize=15,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':15})

axes[3].plot(x2,ema.x,color='black',label='EMA')
axes[3].legend(loc = 'upper right')


axes[4].plot(x2,sg,color='orange',label='SG')
axes[4].legend(loc = 'upper right')

axes[5].plot(x2[start_index:],dn_ann,color='green',label='ANN (m = 3)')
axes[5].legend(loc = 'upper right')






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
    
    
x = np.linspace(-2, 4, 2000)
#y = (x+3) * ((x-2)**2) * (x+1)**3
y = (5 * x**5) - (20*(x**4)) + (5 * (x**3)) + (50 *(x**2) - (20*x)) -40
mean_noise = 0
noise = np.random.normal(mean_noise, 30, len(y))

y_noise = y + noise



def get_features(array, train_ratio = 0.75):
    
    x = []
    y = []

    m = 3 
    
    for i in range(len(array) - m):
        
        x.append(array[i:i+m])############################### Gets 3 day data
        y.append(array[i+m]) 
        
        
    x = np.array(x)
    y = np.array(y)
    
    last_index = int(len(x)*train_ratio)
    
    train_x = x[0:last_index]
    train_y = y[0:last_index]
    test_x = x[last_index:]
    test_y = y[last_index:]
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = get_features(y_noise)

#tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2)
#tpot.fit(train_x, train_y)
#print(tpot.score(test_x, test_y))
#tpot.export('polynomial5.py')


# Average CV score on the training set was:-1266.7900678552883

def clf(in_put, out_put):
    
    exported_pipeline = make_pipeline(
                        MaxAbsScaler(),
                        Nystroem(gamma=0.45, kernel="poly", n_components=10),
                        ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.1)
                    )

    exported_pipeline.fit(in_put, out_put)
    results = exported_pipeline.predict(in_put)
    
    return results

pred_train = clf(train_x, train_y)
pred_test = clf(test_x, test_y)


final = np.concatenate((pred_train,pred_test))
noise = np.concatenate((train_x[:,0],test_x[:,0]))
start_index = len(x) - (len(final))

RMSE_tpot = math.sqrt(mean_squared_error(y[start_index:],final))
fig, ax = plt.subplots()

ax.plot(x[start_index:],noise, label = 'Noise', color=dyna_colors[2])
ax.plot(x[start_index:],final,color=dyna_colors[1], label = 'TPOT')
ax.plot(x,y, linestyle = '-.',color = dyna_colors[0],label = r'$5x^5 - 20x^4 + 5x^3 + 50x^2 - 20x - 40$')
ax.legend(loc = 'upper left', fontsize = 15)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.01, 0.75, 'RMSE = {}'.format(RMSE_tpot), transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
