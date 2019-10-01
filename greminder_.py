# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:39:13 2018
"""

import pandas as pd

df = pd.read_csv('greminder.csv')

def time_delta(msg, oras):
    ts_written = msg['ts_sent']
    ts_written = pd.to_datetime(ts_written)
    ts_time = ts_written.apply(lambda x: x.time() if not pd.isnull(x) else '')
    ts_time = ts_written.apply(lambda x: timedelta(hours = x.hour, minutes = x.minute,
                                              seconds = x.second) if not pd.isnull(x) else '')
    if oras == '7:30':
        thresh_time = time(06,00,00)
        thresh = timedelta(hours = thresh_time.hour, minutes = thresh_time.minute, seconds = thresh_time.second)
    elif oras =='11:30':
        thresh_time = time(10,00,00)
        thresh = timedelta(hours = thresh_time.hour, minutes = thresh_time.minute, seconds = thresh_time.second)
    else:
        thresh_time = time(14,00,00)
        thresh = timedelta(hours = thresh_time.hour, minutes = thresh_time.minute, seconds = thresh_time.second)
    
    exceeded = []
    for x in ts_time:
        if x > thresh:

            exceeded.append(x)
            
    exceeded = np.array(exceeded)
    print thresh_time
    return exceeded, thresh
####################################################################################################

def delay(time_delta, thresh):
    sec = time_delta - thresh
    
    delay = lambda x: x.total_seconds()
    delay_func = np.vectorize(delay)
    delay_sec = delay_func(sec)
    delay_min = delay_sec / 60
    
    return delay_sec, delay_min