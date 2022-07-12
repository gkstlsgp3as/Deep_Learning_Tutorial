#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:38:23 2022

@author: hsh0514

np_avg.py
"""

#%% load libraries

import os
import os.path 
import sys

import numpy as np
import pickle
from netCDF4 import Dataset
import pandas as pd

rootPath = '/home/hsh0514/data/'
colPath = '2021/03/'

#%% define function

def open_netcdf(fname):
    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()

    fid=Dataset(fname,'r', format="NETCDF4")
    print("Open:",fname)
    return fid

#%%
t_var = ['rad','geo','pm','meteo']
m_var = ['ugrd','vgrd','tmp','rh','dpt','visip','hpbl','pres']


for i, d in enumerate(range(1,32)):
    day = str(d).zfill(2)
    fl = os.listdir(rootPath+colPath+day)
    print(day+': '+str(len(fl))+' files')
    
    for j, f in enumerate(fl):
        nc  = open_netcdf(rootPath+colPath+day+'/'+f)
        
        rad = nc.variables['gems_rad'][:]
        geo = nc.variables['gems_geo'][:]
        pm = nc.variables['pm25'][:]
        
        meteo = nc.variables['ugrd'][:]
        for m in m_var[1:]:
                meteo = np.vstack((meteo,nc.variables[m][:]))
        
        meteo = np.transpose(meteo)
        
        if i==0 and j==0: 
            avg = np.array([rad.mean(axis=0), geo.mean(axis=0), meteo.mean(axis=0)]) # 0: row, 1: col
            sd = np.array([rad.std(axis=0), geo.std(axis=0), meteo.std(axis=0)])
        else: 
            avg = np.vstack((avg, np.array([rad.mean(axis=0), geo.mean(axis=0), meteo.mean(axis=0)])))
            sd = np.vstack((sd, np.array([rad.std(axis=0), geo.std(axis=0), meteo.std(axis=0)]))) 
        
        nc.close()
        
with open(rootPath + 'GEMS_z_norm.pkl','wb') as f:
    pickle.dump([avg,sd],f) # dump pickle
