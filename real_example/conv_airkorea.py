#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#conv_airkorea.py
"""
Created on Wed Jan  5 15:51:42 2022

@author: hsh0514
"""

import os
import pandas as pd
import re

dataPath = '/home/hsh0514/PM25/conc/data/'
outPath = '/home/hsh0514/PM25/conc/temp/'

fl = os.listdir(dataPath)

for i in range(len(fl)):
    xlsx = pd.read_excel(dataPath+fl[i])
    date = re.sub(r'[^0-9]', '', fl[i])
    if len(date) == 5: 
        date = date[0:4]+'0'+date[-1:]
    
    xlsx.to_csv(outPath+date+'.csv')
