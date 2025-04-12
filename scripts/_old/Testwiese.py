# -*- coding: utf-8 -*-
"""

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset #Dataset
import xlrd
from datetime import datetime
import time
from dateutil import tz

import requests
#import json
#from datetime import datetime
from dateutil import parser

import matplotlib.pyplot as plt
#from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

def read_model_data(filepath=r'C:\data_local\ML\Modelldaten.xlsb', sheetname='Modelldaten'):
    df = pd.read_excel(filepath,sheetname, usecols='A:AI', nrows=8760)
    #df.insert(0, 'Timestamp_left', 0)
    #df['Timestamp_left']=pd.to_datetime(df[['year','month','day']]) + pd.to_timedelta(df['hour'], 'h')
    df['Timestamp_left']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_left'], 0)),axis=1)
    df['Timestamp_right']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_right'], 0)),axis=1)
    df=df.set_index(['Timestamp_left','Timestamp_right'])
    return df

def read_input_data(filepath=r'C:\data_local\ML\Prognoseersatzwerte.xlsb', sheetname='Inputdaten'):
    df = pd.read_excel(filepath,sheetname, usecols='A:E', nrows=48, dtype='float64', skiprows=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16])
    #df=df.drop(df.columns[1], axis=1)
    df.columns.values[0] = 'Timestamp_left'
    df.columns.values[1] = 'Timestamp_right'
    df['Timestamp_left']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_left'], 0)),axis=1)
    df['Timestamp_right']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_right'], 0)),axis=1)
    df=df.set_index(['Timestamp_left','Timestamp_right'])
    return df

def convert_list_string_to_timestamp(inputlist):
    outputlist = []
    to_zone = tz.gettz('Europe/Vienna')
    for i in inputlist:
        outputlist.append(parser.parse(i).astimezone(to_zone))       
    return outputlist

def get_10min_dataslice(session, parameter='TL', start='2022-01-01T00:00', end='2023-01-01T00:00', station_ids='5882',output_format='geojson'):
    
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
    print("Start Request:", station_ids, parameter, time.strftime("%H:%M:%S", time.localtime()))
    response = session.get(base_url, params=payload)
    print("End Request: ", station_ids, parameter, time.strftime("%H:%M:%S", time.localtime()))
    
    dict_response = response.json()
    #print(dict_response)
    #station = dict_response['features'][0]['properties']['station']
    #description = dict_response['features'][0]['properties']['parameters'][parameter]['name']
    #unit = dict_response['features'][0]['properties']['parameters'][parameter]['unit']
    timestamps = dict_response['timestamps']
    return timestamps

def add_something(a,b):
    c = a+b
    d = a*b
    return c,d

if __name__ == "__main__":
    #s = requests.session()
    #timestamps = get_10min_dataslice(s)
    #print(timestamps[0])
    #timestamps = convert_list_string_to_timestamp(timestamps)
    #to_zone = tz.gettz('Europe/Vienna')
    #timestamps = timestamps.astimezone(to_zone)
    
    startendlist = ('2014-01-01T00:00','2015-01-01T00:00','2016-01-01T00:00','2017-01-01T00:00','2018-01-01T00:00','2019-01-01T00:00','2020-01-01T00:00','2021-01-01T00:00','2022-01-01T00:00','2023-01-01T00:00')
    for index in range(0,len(startendlist)-1):
        print(startendlist[index],startendlist[index+1])
    
    