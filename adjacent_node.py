from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

plt.ion()

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import analysis.querydb as qdb
import dynadb.db as db
import analysis.subsurface.proc as proc
import analysis.subsurface.rtwindow as rtw
#------------------------------------------------------------------------------

def RoundReleaseTime(date_time):
    # rounds time to 4/8/12 AM/PM
    time_hour = int(date_time.strftime('%H'))

    quotient = time_hour / 4
    if quotient == 5:
        date_time = datetime.combine(date_time.date() + timedelta(1), time(0,0,0))
    else:
        date_time = datetime.combine(date_time.date(), time((quotient+1)*4,0,0))
            
    return date_time


def node_disp(df):
    node_kinematics = df.loc[:, ['ts', 'tsm_name', 'node_id', 'vel_xz', 'vel_xy']]
    node_kinematics.loc[:, 'disp_xz'] = df.loc[:, 'xz'] - df.shift(3*24*2).loc[:, 'xz']
    node_kinematics.loc[:, 'disp_xy'] = df.loc[:, 'xy'] - df.shift(3*24*2).loc[:, 'xy']
    node_kinematics.loc[:, ['vel_xz', 'vel_xy', 'disp_xz', 'disp_xy']] = node_kinematics.loc[:, ['vel_xz', 'vel_xy', 'disp_xz', 'disp_xy']].abs()
    return node_kinematics


def tsm_analysis(df, window, sc):
    try:
        end = max(df.ts)
        start = min(df.ts)
        window.end = end
        window.start = start - timedelta(days=3)
        window.offsetstart = window.start - timedelta(days=(sc['subsurface']
                ['num_roll_window_ops']*window.numpts-1)/48.)
    
        tsm_name = df['tsm_name'].values[0]
        print(tsm_name)
        tsm_props = qdb.get_tsm_list(tsm_name)[0]
    
        data = proc.proc_data(tsm_props, window, sc, realtime=True, comp_vel=True).tilt.reset_index() # proc using lowess

        ## check treshold exceedance
        
        percent_movt = pd.merge(df, data, how='inner', on=['ts', 'tsm_name', 'node_id'])
        return percent_movt.loc[:, ['ts', 'tsm_name', 'node_id', 'na_status', 'pred']]
    except:
        pass


def plot_hist(df, column, validity, path):
    percent_array = df[column].values
    hist, bins = np.histogram(percent_array, bins=list(range(0, 101, 1)))
    plt.hist(percent_array, bins=list(range(0, 101, 1)))
    plt.axvline(50, linestyle='-', color='r')
    below = hist[0:50]
    above = hist[50:100]
    plt.annotate(str(np.sum(below)), (40, np.max(hist) - 2))
    plt.annotate(str(np.sum(above)), (52, np.max(hist) - 2))
    plt.title(column + '_' + validity)
    plt.savefig(path + column + '_' + validity + '.png')
    plt.close()


def main(window, sc):

    query  = "SELECT * FROM "
    query += "	node_alerts "
    query += "INNER JOIN "
    query += "	tsm_sensors "
    query += "USING (tsm_id) "
    alert_list = db.df_read(query)
    alert_list = alert_list.dropna(subset=['na_status']).drop_duplicates(['tsm_id', 'node_id', 'ts'])
    

#    alert_list = alert_list[alert_list.tsm_id.isin([2, 60])] #################

    tsm_alerts = alert_list.groupby('tsm_id', as_index=False)
    percent_movt = tsm_alerts.apply(tsm_analysis, window=window, sc=sc).reset_index(drop=True)
    percent_movt.to_csv('percent_movement.csv', index=False)
    
    return percent_movt

###############################################################################

if __name__ == "__main__":
    
    run_start = datetime.now()
    
    window, sc = rtw.get_window()
    percent_movt = pd.read_csv('percent_movement.csv')
    
    valid_movt = percent_movt.loc[percent_movt.na_status == 1, :]
#    invalid_movt = percent_movt[percent_movt.na_status == -1]
#    
#    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
#    plot_path = sc['fileio']['realtime_path']
#    
#    plot_hist(valid_movt, 'top_vel', 'valid', output_path + plot_path)
#    plot_hist(valid_movt, 'top_disp', 'valid', output_path + plot_path)
#    plot_hist(valid_movt, 'bottom_vel', 'valid', output_path + plot_path)
#    plot_hist(valid_movt, 'bottom_disp', 'valid', output_path + plot_path)
#    
#    plot_hist(invalid_movt, 'top_vel', 'invalid', output_path + plot_path)
#    plot_hist(invalid_movt, 'top_disp', 'invalid', output_path + plot_path)
#    plot_hist(invalid_movt, 'bottom_vel', 'invalid', output_path + plot_path)
#    plot_hist(invalid_movt, 'bottom_disp', 'invalid', output_path + plot_path)
    
    valid_movt = valid_movt.dropna(subset=['top_disp', 'bottom_disp', 'top_vel', 'bottom_vel'])
    query = "SELECT site_id, tsm_name FROM tsm_sensors"
    tsm_sensors = db.df_read(query)
    tsm_sensors = tsm_sensors.drop_duplicates('tsm_name')
    valid_movt = pd.merge(valid_movt, tsm_sensors, on=['tsm_name'],
                             validate='m:1')

    valid_below3 = valid_movt.loc[((valid_movt.top_disp <= 3)&(valid_movt.bottom_disp <= 3))]#|((valid_movt.top_vel <= 3)&(valid_movt.bottom_vel <= 3)), :]

    query =  "select pe.event_id, event_start, site_id, data_timestamp as ts, trigger_type, validity from "
    query += "  (select * from public_alert_event "
    query += "  where status != 'invalid' "
    query += "  ) pe "
    query += "inner join "
    query += "  public_alert_release "
    query += "using (event_id) "
    query += "inner join "
    query += "  public_alert_trigger "
    query += "using (release_id) "
    query += "inner join "
    query += "  sites "
    query += "using (site_id)"
    public_alert = db.df_read(query)
    
    for index in valid_below3.index:
        df = valid_below3.loc[valid_below3.index == index, :]
        site_id = df['site_id'].values[0]
        ts = df['ts'].values[0]
        event = public_alert.loc[(public_alert.site_id == site_id)&(public_alert.event_start-timedelta(1) <= ts)&(public_alert.validity > ts), :]
        valid_below3.loc[valid_below3.index == index, 'triggers'] = ''.join(set(event['trigger_type']))
        valid_below3.loc[valid_below3.index == index, 'event_id'] = ','.join(map(str, set(event['event_id'])))
    
    print (valid_below3)
    
    valid_below3 = valid_below3[valid_below3.triggers.str.contains('S|g|G|m|M')]
    
    print ('runtime =', datetime.now() - run_start)