from datetime import datetime, timedelta, time
import os
import pandas as pd
import sys

import alertlib as lib
import proc
import rtwindow as rtw
import trendingalert as trend
import plotterlib as plotter

#include the path of outer folder for the python scripts searching
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1,path)
del path   

import querydb as qdb

def RoundReleaseTime(date_time):
    # rounds time to 4/8/12 AM/PM
    time_hour = int(date_time.strftime('%H'))

    quotient = time_hour / 4
    if quotient == 5:
        date_time = datetime.combine(date_time.date() + timedelta(1), time(0,0,0))
    else:
        date_time = datetime.combine(date_time.date(), time((quotient+1)*4,0,0))
            
    return date_time

def writeOperationalTriggers(site_id, end):

    query =  "SELECT sym.alert_level, trigger_sym_id FROM ( "
    query += "  SELECT alert_level FROM "
    query += "    (SELECT * FROM tsm_alerts "
    query += "    where ts <= '%s' " %end
    query += "    and ts_updated >= '%s' " %end
    query += "    ) as ta "
    query += "  INNER JOIN "
    query += "    (SELECT tsm_id FROM tsm_sensors "
    query += "    where site_id = %s " %site_id
    query += "    ) as tsm "
    query += "  on ta.tsm_id = tsm.tsm_id "
    query += "  ) AS sub "
    query += "INNER JOIN "
    query += "  (SELECT trigger_sym_id, alert_level FROM "
    query += "    operational_trigger_symbols AS op "
    query += "  INNER JOIN "
    query += "    (SELECT source_id FROM trigger_hierarchies "
    query += "    WHERE trigger_source = 'subsurface' "
    query += "    ) AS trig "
    query += "  ON op.source_id = trig.source_id "
    query += "  ) as sym "
    query += "on sym.alert_level = sub.alert_level"
    df = qdb.get_db_dataframe(query)
    
    trigger_sym_id = df.sort_values('alert_level', ascending=False)['trigger_sym_id'].values[0]
        
    operational_trigger = pd.DataFrame({'ts': [end], 'site_id': [site_id], 'trigger_sym_id': [trigger_sym_id], 'ts_updated': [end]})
    
    qdb.alert_to_db(operational_trigger, 'operational_triggers')

def main(tsm_name='', end='', end_mon=False):
    run_start = datetime.now()
    qdb.print_out(run_start)
    qdb.print_out(tsm_name)

    if tsm_name == '':
        tsm_name = sys.argv[1].lower()

    if end == '':
        try:
            end = pd.to_datetime(sys.argv[2])
        except:
            end = datetime.now()
    else:
        end = pd.to_datetime(end)
    
    window, sc = rtw.get_window(end)

    tsm_props = qdb.get_tsm_list(tsm_name)[0]
    data = proc.proc_data(tsm_props, window, sc)
    tilt = data.tilt[window.start:window.end]
    lgd = data.lgd
    tilt = tilt.reset_index().sort_values('ts',ascending=True)
    
    if lgd.empty:
        qdb.print_out('%s: no data' %tsm_name)
        return

    nodal_tilt = tilt.groupby('node_id', as_index=False)     
    alert = nodal_tilt.apply(lib.node_alert, colname=tsm_props.tsm_name, num_nodes=tsm_props.nos, disp=float(sc['subsurface']['disp']), vel2=float(sc['subsurface']['vel2']), vel3=float(sc['subsurface']['vel3']), k_ac_ax=float(sc['subsurface']['k_ac_ax']), lastgooddata=lgd, window=window, sc=sc).reset_index(drop=True)
    
    alert['col_alert'] = -1
    col_alert = pd.DataFrame({'node_id': range(1, tsm_props.nos+1), 'col_alert': [-1]*tsm_props.nos})
    node_col_alert = col_alert.groupby('node_id', as_index=False)
    node_col_alert.apply(lib.column_alert, alert=alert, num_nodes_to_check=int(sc['subsurface']['num_nodes_to_check']), k_ac_ax=float(sc['subsurface']['k_ac_ax']), vel2=float(sc['subsurface']['vel2']), vel3=float(sc['subsurface']['vel3']))

    valid_nodes_alert = alert.loc[~alert.node_id.isin(data.inv)]
    
    if max(valid_nodes_alert['col_alert'].values) > 0:
        pos_alert = valid_nodes_alert[valid_nodes_alert.col_alert > 0]
        site_alert = trend.main(pos_alert, tsm_props.tsm_id, window.end, data.inv)
    else:
        site_alert = max(lib.get_mode(list(valid_nodes_alert['col_alert'].values)))
        
    tsm_alert = pd.DataFrame({'ts': [window.end], 'tsm_id': [tsm_props.tsm_id], 'alert_level': [site_alert], 'ts_updated': [window.end]})

    qdb.alert_to_db(tsm_alert, 'tsm_alerts')
    
    writeOperationalTriggers(tsm_props.site_id, window.end)

#######################

#    query = "SELECT ts, alert_level, alert_type, ts_updated FROM"
#    query += " (SELECT * FROM public_alerts WHERE site_id = %s) AS pub" %tsm_props
#    query += " INNER JOIN public_alert_symbols AS sym"
#    query += " ON pub.pub_sym_id = sym.pub_sym_id"
#    query += " ORDER BY ts DESC LIMIT 1"
#    public_alert = qdb.get_db_dataframe(query)
#    if ((public_alert['alert_level'].values[0] != 0 or \
#            public_alert['alert_type'].values[0] == 'event') \
#            and (str(window.end.time()) in ['07:30:00', '19:30:00'] or end_mon)) \
#            or (public_alert.alert.values[0] == 'A0'
#            and RoundReleaseTime(pd.to_datetime(public_alert.timestamp.values[0])) \
#            == RoundReleaseTime(window.end)):
#    plotter.main(data, tsm_props, window, sc, realtime=False)

#######################

    qdb.print_out(tsm_alert)
    
    qdb.print_out('run time = ' + str(datetime.now()-run_start))

################################################################################

if __name__ == "__main__":
    main()
