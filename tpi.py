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
    node_kinematics.loc[:, 'disp_xz'] = df['xz'] - df.shift(3*24*2)['xz']
    node_kinematics.loc[:, 'disp_xy'] = df['xy'] - df.shift(3*24*2)['xy']
    node_kinematics[['vel_xz', 'vel_xy', 'disp_xz', 'disp_xy']] = node_kinematics[['vel_xz', 'vel_xy', 'disp_xz', 'disp_xy']].abs()
    return node_kinematics


def adj_movt(node_data, data, num_nodes):
    curr_node = node_data['node_id'].values[0]
    adj_node_movt = node_data.loc[:, ['ts', 'tsm_name', 'node_id']]
    # movt of top node
    if curr_node > 1:
        top_data = data[data.node_id == curr_node - 1]
        adj_node_movt.loc[:, 'top_disp'] = np.array(top_data[['disp_xz', 'disp_xy']].max(axis=1)) / np.array(node_data[['disp_xz', 'disp_xy']].replace(0, np.nan).min(axis=1, skipna=True))
        adj_node_movt.loc[:, 'top_disp'] = np.where((adj_node_movt['top_disp'] < 1) & (adj_node_movt['top_disp'] > 0), adj_node_movt['top_disp'], 1/adj_node_movt['top_disp'])
        adj_node_movt.loc[:, 'top_vel'] = np.array(top_data[['vel_xz', 'vel_xy']].max(axis=1)) / np.array(node_data[['vel_xz', 'vel_xy']].replace(0, np.nan).min(axis=1, skipna=True))
        adj_node_movt.loc[:, 'top_vel'] = np.where((adj_node_movt['top_vel'] < 1) & (adj_node_movt['top_vel'] > 0), adj_node_movt['top_vel'], 1/adj_node_movt['top_vel'])
    else:
        adj_node_movt.loc[:, 'top_disp'] = np.nan
        adj_node_movt.loc[:, 'top_vel'] = np.nan
    # movt of bottom node
    if curr_node < num_nodes:
        bottom_data = data[data.node_id == curr_node + 1]
        adj_node_movt.loc[:, 'bottom_disp'] = np.array(bottom_data[['disp_xz', 'disp_xy']].max(axis=1)) / np.array(node_data[['disp_xz', 'disp_xy']].replace(0, np.nan).min(axis=1, skipna=True))
        adj_node_movt.loc[:, 'bottom_disp'] = np.where((adj_node_movt['bottom_disp'] < 1) & (adj_node_movt['bottom_disp'] > 0), adj_node_movt['bottom_disp'], 1/adj_node_movt['bottom_disp'])
        adj_node_movt.loc[:, 'bottom_vel'] = np.array(bottom_data[['vel_xz', 'vel_xy']].max(axis=1)) / np.array(node_data[['vel_xz', 'vel_xy']].replace(0, np.nan).min(axis=1, skipna=True))
        adj_node_movt.loc[:, 'bottom_vel'] = np.where((adj_node_movt['bottom_vel'] < 1) & (adj_node_movt['bottom_vel'] > 0), adj_node_movt['bottom_vel'], 1/adj_node_movt['bottom_vel'])
    else:
        adj_node_movt.loc[:, 'bottom_disp'] = np.nan
        adj_node_movt.loc[:, 'bottom_vel'] = np.nan
    adj_node_movt[['top_disp', 'bottom_disp', 'top_vel', 'bottom_vel']] = adj_node_movt[['top_disp', 'bottom_disp', 'top_vel', 'bottom_vel']].replace(np.inf, np.nan) * 100
    return adj_node_movt


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
    
        data = proc.proc_data(tsm_props, window, sc, realtime=True, comp_vel=True).tilt.reset_index()
        node_data = data.groupby('node_id', as_index=False)
        tsm_kinematics = node_data.apply(node_disp).reset_index(drop=True)
        node_kinematics = tsm_kinematics.groupby('node_id', as_index=False)
        adj_node_movt = node_kinematics.apply(adj_movt, data=tsm_kinematics, num_nodes=tsm_props.nos).reset_index(drop=True)
        
        percent_movt = pd.merge(df, adj_node_movt, how='inner', on=['ts', 'tsm_name', 'node_id'])
        return percent_movt[['ts', 'tsm_name', 'node_id', 'top_disp', 'bottom_disp', 'top_vel', 'bottom_vel', 'na_status']]
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
    percent_movt = main(window, sc) #pd.read_csv('percent_movement.csv')
    
    valid_movt = percent_movt[percent_movt.na_status == 1]
    invalid_movt = percent_movt[percent_movt.na_status == -1]
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    plot_path = sc['fileio']['realtime_path']
    
    plot_hist(valid_movt, 'top_vel', 'valid', output_path + plot_path)
    plot_hist(valid_movt, 'top_disp', 'valid', output_path + plot_path)
    plot_hist(valid_movt, 'bottom_vel', 'valid', output_path + plot_path)
    plot_hist(valid_movt, 'bottom_disp', 'valid', output_path + plot_path)
    
    plot_hist(invalid_movt, 'top_vel', 'invalid', output_path + plot_path)
    plot_hist(invalid_movt, 'top_disp', 'invalid', output_path + plot_path)
    plot_hist(invalid_movt, 'bottom_vel', 'invalid', output_path + plot_path)
    plot_hist(invalid_movt, 'bottom_disp', 'invalid', output_path + plot_path)
    
    print ('runtime =', datetime.now() - run_start)