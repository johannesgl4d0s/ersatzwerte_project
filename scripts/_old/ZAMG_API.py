# -*- coding: utf-8 -*-
"""

"""

"""
https://dataset.api.hub.zamg.ac.at/v1/docs/daten.html
https://dataset.api.hub.zamg.ac.at/v1/openapi-docs#/historical/Historical_Station_Metadata_station_historical__resource_id__metadata_get
Todos:
    1) Dataslices List in Dataframe umwandeln via Funktion (Richtige Zuordnung via Timestamp)
        1.1) If not, Einleseroutine von class dataset in dataframe umwandeln
    2) Schleife die sämtliche Stationen, Parameter und Zeiträume einliest und hierbei die 1.000.000 Grenze berücksichtigt
    3) Dataslices in DF pro Parameter aufteilen
    4) DFs von 10min in 1h Werte umwandeln (SUM + AVG Verdichtung beachten) [3)+4) können beliebige Reihenfolge sein]    
    5) Schreiben auf Einleseexcel für TSM
        5.1) Alternativ, Python Schnittstelle für TSM bauen
"""

import requests
import pandas as pd
from dateutil import parser
from dateutil import tz
import time

def get_stations(session):
    response = session.get("https://dataset.api.hub.zamg.ac.at/v1/station/{mode}/klima-v1-10min/filter?start_date=2013-01-01&end_date=2013-01-01&has_sunshine=true&is_active=true")
    df = pd.DataFrame.from_dict(response.json()['matching_stations'])
    return df

def get_10min_dataslice(session, parameter='TL', start='2023-01-01T00:00', end='2023-01-02T00:00', station_ids='5882',output_format='geojson'):
    
    #url for datarequest
    base_url = "https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min"
    
    #set parameters for request
    payload = {}    
    payload['parameters']=parameter
    payload['start']=start
    payload['end']=end
    payload['station_ids']=station_ids
    payload['output_format']=output_format
    
    #get data for selected parameters
    starttime = time.strftime("%H:%M:%S", time.localtime())
    #print("Start Request:", station_ids, parameter, start, end, time.strftime("%H:%M:%S", time.localtime()))
    response = session.get(base_url, params=payload)
    print("Request:", station_ids, parameter, start, end, "From:", starttime, "To:", time.strftime("%H:%M:%S", time.localtime()))
    
    dict_response = response.json()
    #print(dict_response)
    station = dict_response['features'][0]['properties']['station']
    #description = dict_response['features'][0]['properties']['parameters'][parameter]['name']
    #unit = dict_response['features'][0]['properties']['parameters'][parameter]['unit']
    timestamps = convert_list_string_to_timestamp(dict_response['timestamps'])
    #print(timestamps)
    data = dict_response['features'][0]['properties']['parameters'][parameter]['data']
    #timefrom = timestamps[0]
    #timeto = timestamps[-1]
    header = pd.MultiIndex.from_product([[station],
                                     [parameter]],
                                    names=['station','parameter'])
    df_dataslice = pd.DataFrame(data, columns=header)
    df_dataslice["timestamp"] = pd.DataFrame(timestamps)
    #df_dataslice[station][parameter] = data
    #df_dataslice["station"] = station
    #df_dataslice["description"] = description
    #df_dataslice["unit"] = unit
    df_dataslice = df_dataslice.set_index("timestamp")
    return df_dataslice

def convert_list_string_to_timestamp(inputlist):
    outputlist = []
    to_zone = tz.gettz('Europe/Vienna')
    for i in inputlist:
        outputlist.append(parser.parse(i).astimezone(to_zone))       
    return outputlist
   

if __name__ == "__main__":
    s = requests.session()
    df_stations = get_stations(s)
    df_zamg = pd.DataFrame(columns=["timestamp"])
    df_zamg = df_zamg.set_index("timestamp")
    df_temp = df_zamg
    df_empty = df_zamg
    df_zamg_avg = df_zamg
    df_zamg_sum = df_zamg
    #parameters = ('FFAM','GSX','RF','RR','SH','SO','TB1','TB2','TL')
    parameters_avg = ('FFAM', 'TL')
    parameters_sum = ('RR', 'SO')
    # FFAM AVG
    # RR SUM
    # SO SUM
    # TL AVG
    stations = ('7704', '20212', '5609', '3202', '6305', '16412', '11803', '11104', '5904')
    #stations = ('500','726')
    
    #startendlist = ('2014-01-01T00:00','2015-01-01T00:00','2016-01-01T00:00','2017-01-01T00:00','2018-01-01T00:00','2019-01-01T00:00','2020-01-01T00:00','2021-01-01T00:00','2022-01-01T00:00','2023-01-01T00:00')
    startendlist = ('2023-01-01T00:00','2023-05-01T00:00')
        
    for index in range(0,len(startendlist)-1):
        #print(startendlist[index],startendlist[index+1])
        concat = True
        df_temp = df_empty
        for st in stations:
            for p in parameters_avg:
                #print(index,st,p,concat)
                df_values = get_10min_dataslice(s,parameter=p,start=startendlist[index],end=startendlist[index+1],station_ids=st)
                if index > 0:
                    df_values = df_values.tail(-1)
                if concat:
                    #df_zamg = pd.concat([df_zamg, df_values],ignore_index=True, sort=False)
                    df_temp = pd.concat([df_temp, df_values],ignore_index=False, sort=False)
                    concat = False
                    #print(df_zamg_avg)
                else:
                    df_temp = df_temp.join(df_values)
                    #print(df_zamg_avg)
        df_zamg_avg = pd.concat([df_zamg_avg, df_temp],ignore_index=False, sort=False)
                    
    for index in range(0,len(startendlist)-1):
        #print(startendlist[index],startendlist[index+1])
        concat = True
        df_temp = df_empty         
        for st in stations:
            for p in parameters_sum:        
                df_values = get_10min_dataslice(s,parameter=p,start=startendlist[index],end=startendlist[index+1],station_ids=st)
                if index > 0:
                    df_values = df_values.tail(-1)
                if concat:
                    #df_zamg = pd.concat([df_zamg, df_values],ignore_index=True, sort=False)
                    df_temp = pd.concat([df_temp, df_values],ignore_index=False, sort=False)
                    concat = False
                    #print(df_zamg_avg)
                else:
                    df_temp = df_temp.join(df_values)
                    #print(df_zamg_avg)
        df_zamg_sum = pd.concat([df_zamg_sum, df_temp],ignore_index=False, sort=False)
                
    df_zamg_sum = df_zamg_sum.resample('H').sum()
    df_zamg_avg = df_zamg_avg.resample('H').mean()
    
    df_zamg = df_zamg_sum
    df_zamg = df_zamg.join(df_zamg_avg)
    df_zamg.index = df_zamg.index.tz_localize(None)    
    out_path = r"C:\data_local\ML\Zamg.xlsx"
    df_zamg.to_excel(out_path, sheet_name='Daten_v2')