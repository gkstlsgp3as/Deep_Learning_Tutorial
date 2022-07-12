#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:36:24 2022

@author: hsh0514

read_pm25.py
"""
#%% import libraries

import pandas as pd
import os
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
from geopy.geocoders import Nominatim
import numpy as np
import re
import urllib
import requests
import xml.etree.ElementTree as ET

dataPath = '/home/hsh0514/PM25/conc/temp/'
outPath = '/home/hsh0514/PM25/conc/'
fl = os.listdir(dataPath)

#%% read csv files  
df = pd.read_csv(dataPath+fl[5],sep=",",dtype='unicode', parse_dates=["측정일시"])
df.head()
df.columns

df['측정소명'].unique()
len(df['측정소명'].unique())

for i in range(len(fl)):
    df = pd.read_csv(dataPath+fl[i],sep=",",dtype='unicode', parse_dates=["측정일시"])
    print(fl[i][0:6]+': '+str(len(df['측정소코드'].unique()))+"개 (도시대기: "+str(len(df[df['망']=='도시대기']['측정소코드'].unique()))+\
        "개, 도로변대기: "+str(len(df[df['망']=='도로변대기']['측정소코드'].unique())) + \
            "개, 교외대기: "+str(len(df[df['망']=='교외대기']['측정소코드'].unique())) + \
                "개, 국가배경농도: "+str(len(df[df['망']=='국가배경농도']['측정소코드'].unique())) + ")")


#%% read csv files  
df = df[df['측정일시'].dt.month==3]
df.head()
df.columns

df['측정소명'].unique()
len(df['측정소명'].unique())

for i in range(1,32):
    tmp = df[df['측정일시'].dt.day==i]
    
    print(str(i).zfill(2)+': '+str(len(tmp['측정소코드'].unique()))+"개 (도시대기: "+str(len(tmp[tmp['망']=='도시대기']['측정소코드'].unique()))+\
        "개, 도로변대기: "+str(len(tmp[tmp['망']=='도로변대기']['측정소코드'].unique())) + \
            "개, 교외대기: "+str(len(tmp[tmp['망']=='교외대기']['측정소코드'].unique())) + \
                "개, 국가배경농도: "+str(len(tmp[tmp['망']=='국가배경농도']['측정소코드'].unique())) + ")")
        
#%% preprocessing data in 2020

ind=[]

for i in range(len(fl)):
    if fl[i].startswith('2021'): 
        ind.append(i)

#df = pd.read_csv(dataPath+fl[ind[16]],sep=",",dtype='unicode', parse_dates=["측정일시"]) # 2020 Mar
df = pd.read_csv(dataPath+fl[ind[0]],sep=",",dtype='unicode', parse_dates=["측정일시"]) # 2020 Dec
for i in ind[1:]: 
    temp_df = pd.read_csv(dataPath+fl[i],sep=",",dtype='unicode', parse_dates=["측정일시"])
    df = pd.concat([df, temp_df])

df = df.dropna(axis=0, subset=['PM25']) # drop rows containing nan values for PM2.5
df = df.drop(['Unnamed: 0'], axis='columns') # drop the first column
df = df.reset_index(inplace=False, drop=True)

err = df.loc[pd.to_datetime(df.측정일시, format='%Y%m%d%H', errors='coerce').isnull(), '측정일시']
er_ind = err.index
for i, date in enumerate(err): 
    df['측정일시'][er_ind[i]] = date[0:8]+'00'      

df['측정일시']=pd.to_datetime(df['측정일시'], format="%Y%m%d%H")
#df['측정일시']=pd.to_datetime(df['측정일시'], format="%Y%m%d%H", errors='coerce')
for i, date in enumerate(err): 
    d = df['측정일시'][er_ind[i]]
    d = d + datetime.timedelta(days=1)
    df['측정일시'][er_ind[i]] = d

df.to_csv(outPath+'2021_PM25.csv',mode='w')

#%%
df = pd.read_csv(outPath+'2021_PM25.csv',sep=",",dtype='unicode', parse_dates=["측정일시"])
df['Time']=df['측정일시']
df['측정일시']=pd.to_datetime(df['측정일시'], format="%Y%m%d%H")
df['PM25']=pd.to_numeric(df['PM25'])
df['PM10']=pd.to_numeric(df['PM10'])
df['SO2']=pd.to_numeric(df['SO2'])
df['CO']=pd.to_numeric(df['CO'])
df['O3']=pd.to_numeric(df['O3'])
df=df.drop(['Unnamed: 0'], axis=1)

st_df = pd.read_csv(outPath+'st_loc.csv',sep=",",dtype='unicode')

#%% pre-process
df = df.rename(columns={'지역': 'Region', '망': 'Network', '측정소코드': 'Code', '측정소명': 'Name', '측정일시': 'Date', '주소': 'Address'})
df['Samp'] = np.where(df['Network'] == '도시대기', 0, np.where(df['Network']=='교외대기', 1, 
                          np.where(df['Network'] == '도로변대기', 2, np.where(df['Network'] == '국가배경농도', 3, -1))))

df=df.drop(['Network'], axis=1, inplace=False)
df = df.rename(columns={'Samp':'Network'})

#%% insert coordinates of stations
df_loc = pd.merge(df, st_df, how='left', on=['Code'])
df_loc = df_loc.drop(['Region','Name','Address'], axis=1)

df_loc.to_csv(outPath+'2021_ref.csv',mode='w')

#%%
df = pd.read_csv(outPath+'2021_ref.csv',sep=",",dtype='unicode', parse_dates=["Date"])
df['Date']=pd.to_datetime(df['Date'], format="%Y%m%d%H")
df['PM25']=pd.to_numeric(df['PM25'])
df['PM10']=pd.to_numeric(df['PM10'])
df['SO2']=pd.to_numeric(df['SO2'])
df['CO']=pd.to_numeric(df['CO'])
df['O3']=pd.to_numeric(df['O3'])
df['Lon']=pd.to_numeric(df['Lon'])
df['Lat']=pd.to_numeric(df['Lat'])

df=df.drop(['Unnamed: 0'], axis=1)



#%% split into monthly data
for j in range(1,32):
    temp = df[df['Date'].dt.month == 3]
    temp = temp[temp['Date'].dt.day == j]
    temp = temp.drop_duplicates()
    temp = temp.reset_index(drop=True, inplace=False)
    
    temp['Day'] = str(temp.loc[0,'Date'].year) + str(temp.loc[0, 'Date'].month).zfill(2) + str(temp.loc[0, 'Date'].day).zfill(2)
    temp['Hour'] = temp['Date'].dt.hour
    temp = temp.drop(['Date'], axis=1)

    temp.to_csv(outPath+'2021_ref/'+str(i).zfill(2)+'/202103'+str(j).zfill(2)+'.csv', mode='w')

#%% daily average
day_df=pd.DataFrame()

for st in df['측정소코드'].unique(): 
    st_avg_df=df[df['측정소코드']==st]
    st_avg_df = st_avg_df.set_index('측정일시') 
    samp_df=pd.DataFrame()
    
    samp_df['PM25']=st_avg_df.PM25.resample('1D').mean()
    samp_df['PM10']=st_avg_df.PM10.resample('1D').mean()
    samp_df['SO2']=st_avg_df.SO2.resample('1D').mean()
    samp_df['CO']=st_avg_df.CO.resample('1D').mean()
    samp_df['O3']=st_avg_df.O3.resample('1D').mean()
    samp_df['Region']=st_avg_df.iloc[0,0]
    samp_df['Type']=st_avg_df.iloc[0,1]
    samp_df['Code']=st_avg_df.iloc[0,2]
    samp_df['Name']=st_avg_df.iloc[0,3]
    samp_df['Address']=st_avg_df.iloc[0,10]
    
    day_df=pd.concat([day_df,samp_df])

day_df.reset_index(inplace=True, drop=False)
day_df.rename(columns={'측정일시': 'Date'}, inplace=True)
day_df.to_csv(outPath+'2021_daily.csv',mode='w')

#%% 

df = pd.read_csv(outPath+'2021/03/20210301.csv',sep=",",dtype='unicode', parse_dates=["Date"])
df['측정일시']=pd.to_datetime(df['Date'], format="%Y%m%d%H")
df['PM25']=pd.to_numeric(df['PM25'])
df['PM10']=pd.to_numeric(df['PM10'])
df['SO2']=pd.to_numeric(df['SO2'])
df['CO']=pd.to_numeric(df['CO'])
df['O3']=pd.to_numeric(df['O3'])
df['Mon']=df['Date'].dt.month
df=df.drop(['Unnamed: 0'], axis=1)

#%% plot
#df=df.set_index('Date')

#df_samp = df[df['측정소명']=='중구']
#df_sum = pd.DataFrame()
#df_sum['PM25'] = df_samp.PM25.resample('1D').mean()
#df_sum['PM25'] = df.PM25.resample('1D').mean()
sns.boxplot(x="Mon", y="PM25", data=df)

#plt.plot(df.index, df['PM25'])
#plt.plot(df_sum.index, df_sum['PM25'])
