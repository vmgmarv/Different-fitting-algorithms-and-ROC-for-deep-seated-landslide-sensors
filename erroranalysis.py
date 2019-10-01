# -*- coding: utf-8 -*-
"""
This module profiles the error bands for each node, and evaluates the accumulated errors in the column position. 

The analysis involves finding modal values for a given duration of sensor data, and assumes that the modal values
at the lower and upper range represent the region where noise and actual data cannot be resolved. 
The modal values are represented by peaks in the distribution of the node values, 
approximated by a gaussian kde. Arbitrary parameters of peak height and area under the curve are used to determine 
whether a peak is signficant or not.  
"""
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
import pandas as pd
import numpy as np


def max_min(df, num_nodes, mx_mn_df):

    m = df.node_id.values[0]

    df_index = mx_mn_df.loc[mx_mn_df.index == m].index[0]
    
    try:
        #processing XZ axis
        z = df.xz.values
        z = z[np.isfinite(z)]
            
        kde = gaussian_kde(z)
        xi_z = np.linspace(z.min()-2*(z.max()-z.min()),z.max()+2*(z.max()-z.min()),1000)
        yi_z = kde(xi_z)
        xm_z, ym_z = find_spline_maxima(xi_z,yi_z)

        #processing XY axis
        y = df.xy.values
        y = y[np.isfinite(y)]
        
        kde = gaussian_kde(y)
        xi_y = np.linspace(y.min()-2*(y.max()-y.min()),y.max()+2*(y.max()-y.min()),1000)
        yi_y = kde(xi_y)
        xm_y, ym_y = find_spline_maxima(xi_y,yi_y)
    
        #assigning maximum and minimum positions of xz and xy            
        try:
            #multimodal
            mx_mn_df.loc[df_index] = [xm_z.max(), xm_z.min(), xm_y.max(), xm_y.min()]

        except:
            #unimodal
            mx_mn_df.loc[df_index] = [xm_z, xm_z, xm_y, xm_y]
           
    except:
        #no data for current node or NaN present in current node
        try:
            mx_mn_df.loc[df_index] = [z[0], z[0], y[0], y[0]]
        except:
            mx_mn_df.loc[df_index] = [0, 0, 0, 0]

    return mx_mn_df

def same_bounds(df):
    if df['xz_maxlist'].values[0] == df['xz_minlist'].values[0]:
        df['xz_maxlist'] = np.nan
        df['xz_minlist'] = np.nan
    if df['xy_maxlist'].values[0] == df['xy_minlist'].values[0]:
        df['xy_maxlist'] = np.nan
        df['xy_minlist'] = np.nan
    return df

def cml_noise_profiling(df, sc, num_nodes):
#==============================================================================
#     description
#     determines peak/s in data distribution to characterize noise, 
#     and computes for accumulated error for the column position due to the noise
#     
#     inputs
#     xz,xy   - dataframe containing xz and xy positions
#     
#     outputs
#     xz_peaks, xy_peaks
#         - list of arrays containing the bounds of the detected noise per node
#     xz_maxlist_cml, xz_minlist_cml, xy_maxlist_cml,xy_minlist_cml 
#         - list of arrays containing cumulative maximum and minimum column positions
#==============================================================================
    
    df2 = df

    if sc['subsurface']['column_fix'] == 'top':
        df2[['xz', 'xy']] = df2[['xz', 'xy']].apply(lambda x: -x)
        

    #initializing maximum and minimum positions of xz and xy
    mx_mn_df = pd.DataFrame({'xz_maxlist': [0]*num_nodes, 'xz_minlist': [0]*num_nodes, 'xy_maxlist': [0]*num_nodes, 'xy_minlist': [0]*num_nodes}, index = range(1, num_nodes+1))
    mx_mn_df = mx_mn_df[['xz_maxlist', 'xz_minlist', 'xy_maxlist', 'xy_minlist']]
    nodal_df = df2.groupby('node_id')
    max_min_df = nodal_df.apply(max_min, num_nodes = num_nodes, mx_mn_df = mx_mn_df)
    max_min_df = max_min_df.reset_index().loc[max_min_df.reset_index().node_id == 1][['level_1', 'xz_maxlist', 'xz_minlist', 'xy_maxlist', 'xy_minlist']].rename(columns = {'level_1': 'node_id'}).set_index('node_id')
    
    if sc['subsurface']['column_fix'] == 'top':
        for_cml = max_min_df.sort_index(ascending = True)
    else:
        for_cml = max_min_df.sort_index(ascending = False)
    max_min_cml = for_cml.cumsum()    
    if sc['subsurface']['column_fix'] == 'top':
        max_min_cml = max_min_cml.sort_index(ascending = False)
            
    max_min_df = max_min_df.reset_index()
    nodal_max_min = max_min_df.groupby('node_id')
    max_min_df = nodal_max_min.apply(same_bounds)
    max_min_df = max_min_df.set_index('node_id')

    return  max_min_df, max_min_cml
    
def find_spline_maxima(xi,yi,min_normpeak=0.05,min_area_k=0.05):
#==============================================================================
#     description
#     extracts peaks from the gaussian_kse function,
#     such that peaks have a minimum normalized peak height and a minimum bound area     
#     
#     inputs
#     xi,yi           points corresponding to the gaussian_kde function of the data
#     min_normpeak    minimum normalized peak of the gaussian_kde function
#     min_area_k      proportional constant multiplied to the maximum bound area to compute the minimum bound area    
#     
#     output
#     peaks[x,y]      the peak locations [x] and heights [y] of the gaussian_kde function
#==============================================================================
    
    #setting gaussian_kde points as spline    
    s0=UnivariateSpline(xi,yi,s=0)
    
    try:
        #first derivative (for gettinge extrema)        
        dy=s0(xi,1)
        s1=UnivariateSpline(xi,dy,s=0)
        
        #second derivative (for getting inflection points)
        dy2=s1(xi,1)
        s2=UnivariateSpline(xi,dy2,s=0)
        
        #solving for extrema, maxima, and inflection points
        extrema=s1.roots()
        maxima=np.sort(extrema[(s2(extrema)<0)])
        inflection=np.sort(s2.roots())
        
        try:
            #setting up dataframe for definite integrals with inflection points as bounds            
            df_integ=pd.DataFrame()
            df_integ['lb']=inflection[:-1]
            df_integ['ub']=inflection[1:]

            #assigning maxima to specific ranges
            df_integ['maxloc']=np.nan            
            for i in range(len(df_integ)):
                try:
                    len((maxima>df_integ['lb'][i])*(maxima<df_integ['ub'][i]))>0
                    df_integ['maxloc'][i]=maxima[(maxima>df_integ['lb'][i])*(maxima<df_integ['ub'][i])]
                except:
                    continue
            
            #filtering maxima based on peak height and area
            df_integ.dropna(inplace=True)
            df_integ['maxpeak']=s0(df_integ['maxloc'])
            df_integ=df_integ[df_integ['maxpeak']>0.001]
            df_integ['normpeak']=s0(df_integ['maxpeak'])/s0(df_integ['maxpeak'].values.max())
            df_integ['area']=df_integ.apply(lambda x: s0.integral(x[0],x[1]), axis = 1)
            df_integ=df_integ[(df_integ['area']>min_area_k*df_integ['area'].values.max())*(df_integ['normpeak']>min_normpeak)]
            return df_integ['maxloc'],df_integ['maxpeak']#,inflection[df_integ.index],s0(inflection[df_integ.index])

        except:
            #filtering maxima based on peak height only
            maxima=extrema[(s2(extrema)<0)*(s0(extrema)/s0(extrema).max()>min_normpeak)]
            return maxima,s0(maxima)#,inflection,s0(inflection)
    
    except:
        #unimodal kde
        return xi[np.argmax(yi)],yi.max()#,None,None
