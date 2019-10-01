# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:52:24 2019
"""

from datetime import timedelta
import numpy as np
import os
import pandas as pd
from statsmodels.formula.api import ols
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess
import filterdata as filt
import erroranalysis as err

#include the path of "Analysis" folder for the python scripts searching
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1,path)
del path   

import querydb as qdb

class ProcData:
    def __init__ (self, invalid_nodes, tilt, lgd, max_min_df, max_min_cml):
        self.inv = invalid_nodes
        self.tilt = tilt
        self.lgd = lgd
        self.max_min_df = max_min_df
        self.max_min_cml = max_min_cml
        
def resample_node(df, window):
    blank_df = pd.DataFrame({'ts': [window.end, window.offsetstart],
                    'node_id': [df['node_id'].values[0]]*2,
                    'tsm_name': [df['tsm_name'].values[0]]*2}).set_index('ts')
    df = df.append(blank_df)
    df = df.reset_index().drop_duplicates(['ts', 'node_id']).set_index('ts')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending = True)
    df = df.resample('30Min').pad()
    df = df.reset_index(level=1)
    return df
      
def no_initial_data(df, num_nodes, offsetstart):
    allnodes = np.arange(1,num_nodes+1)
    w_init_val = df[df.ts < offsetstart+timedelta(hours=0.5)]['node_id'].values
    no_init_val = allnodes[np.in1d(allnodes, w_init_val, invert=True)]
    return no_init_val

def no_data(df, num_nodes):
    allnodes = np.arange(1, num_nodes+1)
    withval = sorted(set(df.node_id))
    noval = allnodes[np.in1d(allnodes, withval, invert=True)]
    return noval

def accel_to_lin_xz_xy(monitoring, seg_len):

    #DESCRIPTION
    #converts accelerometer data (xa,ya,za) to corresponding tilt expressed as
    #horizontal linear displacements values, (xz, xy)
    
    #INPUTS
    #seg_len; float; length of individual column segment
    #xa,ya,za; array of integers; accelerometer data (ideally, -1024 to 1024)
    
    #OUTPUTS
    #xz, xy; array of floats; horizontal linear displacements along the planes 
    #defined by xa-za and xa-ya, respectively; units similar to seg_len
    
    xa = monitoring.x.values
    ya = monitoring.y.values
    za = monitoring.z.values

    theta_xz = np.arctan(za / (np.sqrt(xa**2 + ya**2)))
    theta_xy = np.arctan(ya / (np.sqrt(xa**2 + za**2)))
    xz = seg_len * np.sin(theta_xz)
    xy = seg_len * np.sin(theta_xy)
    
    monitoring['xz'] = np.round(xz,4)
    monitoring['xy'] = np.round(xy,4)
    
    return monitoring

def low_ess(y,x,fraction):
    filtered = lowess(y, x, frac = fraction)
    
    return filtered

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
        xz=df.xz
        lws = low_ess(xz,np.arange(len(xz)), fraction = 0.1)
        lws = lws[:,1]
        lws = pd.Series(lws)
#        df = df.rolling(window=roll_window_numpts, min_periods=1).mean()
#        df = df[roll_window_numpts-1:]
#        return np.round(df, 4)
        return lws
    else:
        return df
        
def node_inst_vel(filled_smoothened, roll_window_numpts, start):
    try:          
        lr_xz = ols(y=filled_smoothened.xz, x=filled_smoothened.td,
                    window=roll_window_numpts, intercept=True)
        lr_xy = ols(y=filled_smoothened.xy, x=filled_smoothened.td,
                    window=roll_window_numpts, intercept=True)
                
        filled_smoothened = filled_smoothened.loc[filled_smoothened.ts >= start]
        
        vel_xz = lr_xz.beta.x.values[0:len(filled_smoothened)]
        vel_xy = lr_xy.beta.x.values[0:len(filled_smoothened)]
        filled_smoothened['vel_xz'] = np.round(vel_xz, 4)
        filled_smoothened['vel_xy'] = np.round(vel_xy, 4)
    
    except:
        qdb.print_out("ERROR in computing velocity")
        filled_smoothened['vel_xz'] = np.zeros(len(filled_smoothened))
        filled_smoothened['vel_xy'] = np.zeros(len(filled_smoothened))
    
    return filled_smoothened

#get_last_good_data(df):
#    evaluates the last good data from the input df
#    
#    Parameters:
#        df: dataframe object
#            input dataframe object where the last good data is to be evaluated
#        
#    Returns:
#        dflgd: dataframe object
#            dataframe object of the resulting last good data
def get_last_good_data(df):
    if df.empty:
        qdb.print_out("Error: Empty dataframe inputted")
        return
    # groupby node_id
    dfa = df.groupby('node_id')
    # extract the latest timestamp per node_id, drop the index
    dflgd =  dfa.apply(lambda x: x[x.ts == x.ts.max()])
    dflgd = dflgd.reset_index(level=1, drop=True)
    
    return dflgd

def proc_data(tsm_props, window, sc, realtime=False, comp_vel=True,
              analysis=True):
    
    monitoring = qdb.get_raw_accel_data(tsm_name=tsm_props.tsm_name,
                from_time=window.offsetstart, to_time=window.end,
                analysis=analysis)
    monitoring = monitoring.loc[monitoring.node_id <= tsm_props.nos]

    monitoring = filt.apply_filters(monitoring)

    #identify the node ids with no data at start of monitoring window
    no_init_val = no_initial_data(monitoring,tsm_props.nos,window.offsetstart)
    
    #get last good data prior to the monitoring window (LGDPM)
    if len(no_init_val) != 0:
        lgdpm = qdb.get_single_lgdpm(tsm_props.tsm_name, no_init_val,
                                     window.offsetstart, analysis=analysis)
        lgdpm = filt.apply_filters(lgdpm)
        lgdpm = lgdpm.sort_index(ascending = False).drop_duplicates('node_id')
        
        monitoring=monitoring.append(lgdpm)

    invalid_nodes = qdb.get_node_status(tsm_props.tsm_id)
    monitoring = monitoring.loc[~monitoring.node_id.isin(invalid_nodes)]

    lgd = get_last_good_data(monitoring)

    #assigns timestamps from LGD to be timestamp of offsetstart
    monitoring.loc[(monitoring.ts<window.offsetstart)|(pd.isnull(monitoring.ts)),
                   ['ts']] = window.offsetstart
    
    monitoring = accel_to_lin_xz_xy(monitoring, tsm_props.seglen)
    
    monitoring = monitoring.drop_duplicates(['ts', 'node_id'])
    monitoring = monitoring.set_index('ts')
    monitoring = monitoring[['tsm_name', 'node_id', 'xz', 'xy']]

    nodes_noval = no_data(monitoring, tsm_props.nos)
    nodes_nodata = pd.DataFrame({'tsm_name': [tsm_props.tsm_name]*len(nodes_noval),
                        'node_id': nodes_noval, 'xy': [np.nan]*len(nodes_noval),
                        'xz': [np.nan]*len(nodes_noval),
                         'ts': [window.offsetstart]*len(nodes_noval)})
    nodes_nodata = nodes_nodata.set_index('ts')
    monitoring = monitoring.append(nodes_nodata)
    
    max_min_df, max_min_cml = err.cml_noise_profiling(monitoring, sc, tsm_props.nos)
        
    #resamples xz and xy values per node using forward fill
    monitoring = monitoring.groupby('node_id').apply(resample_node,
                         window = window).reset_index(level=1).set_index('ts')
    
    nodal_proc_monitoring = monitoring.groupby('node_id')
    
    if not realtime:
        to_smooth = int(sc['subsurface']['to_smooth'])
        to_fill = int(sc['subsurface']['to_fill'])
    else:
        to_smooth = int(sc['subsurface']['rt_to_smooth'])
        to_fill = int(sc['subsurface']['rt_to_fill'])
    
    filled_smoothened = nodal_proc_monitoring.apply(fill_smooth,
                            offsetstart=window.offsetstart, end=window.end,
                            roll_window_numpts=window.numpts, to_smooth=to_smooth,
                            to_fill=to_fill)
    filled_smoothened = filled_smoothened[['xz', 'xy','tsm_name']].reset_index()
       
    if comp_vel == True:
        filled_smoothened['td'] = filled_smoothened.ts.values - \
                                            filled_smoothened.ts.values[0]
        filled_smoothened['td'] = filled_smoothened['td'].apply(lambda x: x / \
                                            np.timedelta64(1,'D'))
        
        nodal_filled_smoothened = filled_smoothened.groupby('node_id') 
        
        tilt = nodal_filled_smoothened.apply(node_inst_vel,
                                            roll_window_numpts=window.numpts,
                                            start=window.start)
        tilt = tilt.reset_index(drop=True)
        tilt = tilt.drop(['td'], axis=1)
        tilt = tilt.set_index('ts')
        tilt = tilt.sort_values('node_id', ascending=True)
    else:
        tilt = filled_smoothened.set_index('ts')
    
    return ProcData(invalid_nodes,tilt.sort_index(),lgd,max_min_df,max_min_cml)
