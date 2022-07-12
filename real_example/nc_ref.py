#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:55:29 2022

@author: hsh0514
nc_ref.py
"""

#%%
#from scipy.io import netcdf
import xarray as xr
import os
import sys
import pickle
import pandas as pd

from netCDF4 import Dataset
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import basemap #conda install -c conda-forge basemap
import numpy as np
from numpy import asarray
from sklearn.metrics import mean_squared_error
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


from tensorflow.keras.layers import Flatten, MaxPooling2D, InputLayer, Conv2D, Dense, Activation, BatchNormalization, Dropout, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, r2_score

from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from imblearn.over_sampling import *

#import PY.optimize as op
import random
import scipy as sp
import scipy.stats
import itertools

rootPath = '/home/hsh0514/data/'
colPath = '2021/03/'
dlPath = '/home/hsh0514/DL/'

#%% define function

def open_netcdf(fname):
    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()

    fid=Dataset(fname,'r', format="NETCDF4")
    print("Open:",fname)
    return fid

def ZNormScaler(data, avg, std):
    return (data-avg) / std

def choice_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1004, idx=0):
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num
    X_train = 0
    X_test = 0
    y_train = 0
    y_test = 0
    
    if shuffle:
        np.random.seed(random_state)
        
        train_idx = np.random.choice(X.shape[0], train_num, replace=False)
        test_idx = np.setdiff1d(asarray(range(0,X.shape[0])),train_idx)
        
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]      
        
    else:
        X_train = X[:train_num]
        X_test = X[train_num:]
        y_train = y[:train_num]
        y_test = y[train_num:]
        
    print(test_idx)

    return X_train, X_test, y_train, y_test, train_idx, test_idx

# for normalize
scaling = True
avg, std = pickle.load(open(rootPath+'GEMS_z_norm.pkl','rb'))

#%% Deep Learning: ANN

#%% 1. NN SETTING 

stop_steps      = 10     # No improvement within the stop_steps -> Terminated
n_epoch         = [20, 50, 100, 200]
val_percent     = 0.2    # 80%: Test, 20%: Validation

act_func        = ['relu']#,'tanh','sigmoid','linear']
act_idx         = 0

nn_list = ['ANN', 'CNN']
n_dim = 100

bt_list         = [64,128,256]
lr_list         = [0.001, 0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
op_list         = ['Adam']#,'SGD']#],'RMSprop']
nd_list         = sp_randint(50,150)#[50, 120]#: not good 

#%% 2. DATA PREPARATION
t_var = ['rad','geo','st','pm','wv','meteo']
m_var = ['ugrd','vgrd','tmp','rh','dpt','visip','hpbl','pres']
g_var = ['longitude','latitude','station']
wv_list = np.hstack((np.arange(310, 324, step=0.2), np.arange(328.6, 365.4, step=0.2), 
           np.arange(432, 450, step=0.2), 388, 412))#, 

for i, d in enumerate(range(1,32)):
    day = str(d).zfill(2)
    fl = os.listdir(rootPath+colPath+day)
    print(day+': '+str(len(fl)))
    
    for j, f in enumerate(fl):
        nc  = open_netcdf(rootPath+colPath+day+'/'+f)
        
        rad = nc.variables['gems_rad'][:]
        geo = nc.variables['gems_geo'][:]
        wv = nc.variables['wavelength'][:]
        wv = list(map(lambda x: round(x,1), wv))
        pm = nc.variables['pm25'][:]
        
        st = nc.variables['longitude'][:]
        for g in g_var[1:]:
            st = np.vstack((st, nc.variables[g][:]))

        meteo = nc.variables['ugrd'][:]
        for m in m_var[1:]:
            meteo = np.vstack((meteo,nc.variables[m][:]))
            
        meteo = np.transpose(meteo)
        
        day_tr = np.full((rad.shape[0], 1), int(f[6:8]))
        hr_tr = np.full((rad.shape[0], 1), int(f[8:10]))
        
        # standardization - delete outliers - z-normalization
        rad_tr = np.copy(rad)
        geo_tr = np.copy(geo)
        meteo_tr = np.copy(meteo)
        
        for l, w in enumerate(wv_list):
            ind = np.where(wv == round(w,1))
            
            tmp = np.array([np.copy(rad[:, ind[0][0]])])
            if l==0: 
                X_raw_tmp = np.copy(tmp)
            else: 
                X_raw_tmp = np.vstack((X_raw_tmp, tmp))
            
        X_raw_tmp = np.vstack((st, day_tr.T, hr_tr.T, X_raw_tmp, geo.T, meteo.T)) 

        Y_tmp = np.copy(pm)
        
        if i+j==0:
            Y_raw = np.copy(Y_tmp)
            X_raw = np.copy(X_raw_tmp)
            
        else: 
            Y_raw = np.hstack((Y_raw, Y_tmp))
            X_raw = np.hstack((X_raw, X_raw_tmp))

        nc.close()
        
#X_tr.dump(dlPath+'ref_data')
#Y_tr.dump(dlPath+'ref_pm_data') # pm_cl_data
#X_raw.dump(dlPath+'raw_data')

#%% 3. PRE-PROCESSING DATA

X_tr = X_raw
Y_tr = Y_raw
for col in range(X_tr.shape[0])[5:]: 
    
    q3, q1 = np.percentile(X_tr[col,:], [75, 25])
    iqr = q3-q1
    
    idx = np.where(X_tr[col,:] < q1-3*iqr)[0]
    idx=np.concatenate((idx, np.where(X_tr[col,:] > q3+3*iqr)[0]))
    print(len(idx))

    Y_tr=np.delete(Y_tr, idx, axis=0)
    X_tr=np.delete(X_tr, idx, axis=1)
    X_tr[col,:] = ZNormScaler(X_tr[col,:], X_tr[col,:].mean(axis=0), X_tr[col,:].std(axis=0))

#%%

#fig, axes = plt.subplots(2,1, figsize=(8,5), constrained_layout=True)

#axes[0].hist(Y_raw, 150)    
#axes[1].hist(Y_tr, 150)

plt.hist(Y_raw, 150, label='original')
plt.hist(Y_tr, 150, label='pre-processed')

plt.legend()
plt.title("Comparison between original and pre-processed data")
    
#%% 4. PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=90)
fit = pca.fit(X_tr)
X_tr_in = pca.fit_transform(X_tr[5:,:].T)

print("Explained Variance: %s" % fit.explained_variance_ratio_)
#print(fit.components_)
print(sum(pca.explained_variance_ratio_))

X_tr_in = np.hstack((X_tr[:5,:].T, X_tr_in))

#%% 5. DATA SPLIT
X_train, X_rem, y_train, y_rem, train_idx, rem_idx = choice_train_test_split(X_tr_in, Y_tr, test_size=0.2, shuffle=True, random_state=1004)

x_val, x_test, y_val, y_test, val_idx, test_idx = choice_train_test_split(X_rem, y_rem, test_size=0.5, shuffle=True, random_state=1004, idx=rem_idx)

test_idx = rem_idx[test_idx]
val_idx = rem_idx[val_idx]
x_val = X_tr_in[val_idx,:]
y_val = Y_tr[val_idx]
x_test = X_tr_in[test_idx,:]
y_test = Y_tr[test_idx]
#%% 6. SAVING DATA

X_tr.dump(dlPath+'ref_data')
Y_tr.dump(dlPath+'ref_pm_data') # pm_cl_data
X_raw.dump(dlPath+'raw_data')
test_idx.dump(dlPath+'test_idx')
train_idx.dump(dlPath+'train_idx')
#X_train_over.dump(dlPath+'train_over_data')
#y_train_over.dump(dlPath+'train_over_pm')
X_train.dump(dlPath+'train_data')
y_train.dump(dlPath+'train_pm')
x_val.dump(dlPath+'val_data')
x_test.dump(dlPath+'test_data')
y_val.dump(dlPath+'val_pm')
y_test.dump(dlPath+'test_pm')

#%% ((Option. Oversampling))
import smogn

X_df = pd.DataFrame(X_train)
X_df['PM']=y_train

X_over = smogn.smoter(data = X_df, y='PM')

asarray(X_df.iloc[:,:-1]).dump(dlPath+'over_data')
asarray(X_df.iloc[:,-1]).dump(dlPath+'over_pm')

#%% 7. DATA LOADING

X_raw = np.load(dlPath+'raw_data', allow_pickle=True)
X_tr = np.load(dlPath+'ref_data', allow_pickle=True)
Y_tr = np.load(dlPath+'ref_pm_data', allow_pickle=True)

X_train = np.load(dlPath+'train_data', allow_pickle=True)
y_train = np.load(dlPath+'train_pm', allow_pickle=True)
X_train_over = np.load(dlPath+'over_data', allow_pickle=True)
y_train_over = np.load(dlPath+'over_pm', allow_pickle=True)
x_val = np.load(dlPath+'val_data', allow_pickle=True)
x_test = np.load(dlPath+'test_data', allow_pickle=True)
y_val = np.load(dlPath+'val_pm', allow_pickle=True)
y_test = np.load(dlPath+'test_pm', allow_pickle=True)
test_idx = np.load(dlPath+'test_idx', allow_pickle=True)
train_idx = np.load(dlPath+'train_idx', allow_pickle=True)

#%% 8. MODEL BUILDING AND RUNNING
nn_type = nn_list[1]
opt_name = op_list[0]
lr = lr_list[0]
batch_size = bt_list[0]

#%% DEEP LEARNING
## ANN
## HYPERPARAMETERS

X_train = X_train_over
y_train = y_train_over

def ANN(optimizer = 'adam',nodes=100,batch_size=64,epochs=50,activation='relu',patience=5,loss='mse', learning_rate=0.01):
    
    if (opt_name == 'Adam'):
        optimizer_in = tf.optimizers.Adam(learning_rate=learning_rate)
    elif (opt_name == 'SGD'):
        optimizer_in = tf.optimizers.SGD(learning_rate=learning_rate)
    elif (opt_name == 'RMSprop'):
        optimizer_in = tf.optimizers.RMSprop(learning_rate=learning_rate)
    
    model = Sequential()
    model.add(Dense(nodes, input_shape=(x_val[:,5:].shape[1],), activation=activation))
    model.add(Dense(nodes, activation=activation))
    model.add(Dense(units=1)) #y_train.size
    model.compile(optimizer=optimizer_in,loss=loss, metrics=['mae', 'mse'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0, patience=patience,verbose=0,mode='min')
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience = patience)
    model.summary()
    
    history = model.fit(x_val[:,5:], y_val,
              batch_size=batch_size,
              epochs=epochs,
              callbacks = [early_stop],
              verbose=1) #verbose set to 1 will show the training process
    nn_true = True
    return model

def CNN(nodes=50, filter_num=64, filter_size=3, stride=2, batch_size=64, activation='relu', patience=5, epochs=50, learning_rate=0.01, loss='mse'):

    if (opt_name == 'Adam'): optimizer_in = tf.optimizers.Adam(learning_rate=learning_rate)
    elif (opt_name == 'SGD'):
        optimizer_in = tf.optimizers.SGD(learning_rate=learning_rate)
    elif (opt_name == 'RMSprop'):
        optimizer_in = tf.optimizers.RMSprop(learning_rate=learning_rate)
    
    l2_loss_lambda = 0.010
    l2 = regularizers.l2(l2_loss_lambda)
    dim = x_val[:,5:].shape
    #x_val_in = x_val[:,3:].reshape(-1)#,dim[1],1, 1)
    model = Sequential()
# Input layer
    #model.add(InputLayer(dim))
# Hidden layer
    model.add(Conv1D(filter_num,activation=activation,kernel_size=filter_size, 
                     padding='same', strides=stride, input_shape=(dim[1],1), kernel_regularizer=l2))
    
    model.add(Dropout(0.2)) #20% of the nodes are set to 0 to prevent overfitting
    model.add(Conv1D(filter_num, kernel_size=filter_size, activation='relu', padding='valid', kernel_regularizer=l2))
    
    model.add(MaxPooling1D(pool_size=stride)) # Normally, pooling size is equal to the stride
    model.add(Flatten())
    model.add(Dense(units=nodes, activation='relu')) 
    
    model.add(Dropout(0.3))
    model.add(Dense(units=1)) 
    
    model.compile(optimizer=optimizer_in,loss=loss, metrics=['mae', 'mse'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0, patience=patience,verbose=0,mode='min')
    model.summary()
    
    history = model.fit(x_val[:,5:], y_val,
              batch_size=batch_size,
              epochs=epochs,
              callbacks = [early_stop],
              verbose=1) #verbose set to 1 will show the training process
    
    nn_true = True
    return model
    
n_iter = 10

if (nn_type == 'ANN'):
    lr = []
    for i in range(0, n_iter):
        n = 10 ** np.random.uniform(-4,-1)
        lr.append(n)
    
    params = {
        'activation': act_func,
        'batch_size': bt_list, 
        'nodes': sp_randint(50,200), 
        'epochs': n_epoch, 
        'learning_rate': lr, 
        'patience': random.sample(range(3,20),n_iter)
        }
    
    clf = KerasRegressor(build_fn=ANN, verbose=0)
elif (nn_type == 'CNN'):
    lr = []
    for i in range(0, n_iter):
        n = 10 ** np.random.uniform(-4,-1)
        lr.append(n)
    
    params = {
        'activation': act_func,
        'batch_size': bt_list, 
        'nodes': sp_randint(50,200), 
        'epochs': n_epoch, 
        'learning_rate': lr, 
        'patience': random.sample(range(3,20),n_iter),
        'filter_size': [3,5,7,9],
        'filter_num': bt_list,
        'stride': [1,2,3]
        }
    
    clf = KerasRegressor(build_fn=CNN, verbose=0)
    
Random = RandomizedSearchCV(clf, param_distributions=params, cv=3, 
                            n_iter=n_iter, scoring='neg_mean_squared_error')
Random.fit(x_val[:,5:], y_val)
print(Random.best_params_)
print("MSE:" + str(-Random.best_score_))

prms = Random.best_params_

if (nn_type == 'ANN'):
    model = Sequential()
    model.add(Dense(prms['nodes'], input_shape=(X_train[:,5:].shape[1],), activation=prms['activation']))
    model.add(Dense(prms['nodes'], activation=prms['activation']))
    model.add(Dense(units=1)) #y_train.size
    
    nn_true = True
    
elif (nn_type == 'CNN'):
    dim = X_train[:,5:].shape
    
    l2_loss_lambda = 0.010
    l2 = regularizers.l2(l2_loss_lambda)
    
    model = Sequential()
    
    model.add(Conv1D(prms['filter_num'],activation=prms['activation'],kernel_size=prms['filter_size'], 
                     padding='same', strides=prms['stride'], input_shape=(dim[1],1), kernel_regularizer=l2))
    
    model.add(Dropout(0.2)) #20% of the nodes are set to 0 to prevent overfitting
    model.add(Conv1D(prms['filter_num'], kernel_size=prms['filter_size'], activation=prms['activation'], padding='valid', kernel_regularizer=l2))
    
    model.add(MaxPooling1D(pool_size=prms['stride'])) # Normally, pooling size is equal to the stride
    model.add(Flatten())
    model.add(Dense(units=prms['nodes'], activation=prms['activation'])) 
    
    model.add(Dropout(0.3))
    model.add(Dense(units=1)) 
    
    nn_true = True

if (nn_true):
    if (opt_name == 'Adam'):
        optimizer_in = tf.optimizers.Adam(learning_rate=prms['learning_rate'])
    elif (opt_name == 'SGD'):
        optimizer_in = tf.optimizers.SGD(prms['learning_rate'])
    elif (opt_name == 'RMSprop'):
        optimizer_in = tf.optimizers.RMSprop(prms['learning_rate'])
                            
    model.compile(optimizer=optimizer_in,loss='mse', metrics=['mae', 'mse'])
            
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0, 
                                                  patience=prms['patience'],verbose=0,mode='min')
    check_path = dlPath + '/TEMP/checkpoint'
    iteration = True ; num_itr = 0
#    while iteration:           
    mc = tf.keras.callbacks.ModelCheckpoint(filepath=check_path,monitor='loss',mode='min',
                                            save_best_only=True,save_weights_only=True)

    result = model.fit(X_train[:,5:], y_train, epochs=prms['epochs'],shuffle=True,verbose=1,callbacks=[early_stop,mc])#,
                       #steps_per_epoch=len(X_train[:,5:])//prms['batch_size'])
        
    temp = result.history['loss']
    model.load_weights(check_path)
    #num_itr +=  1
        # if ((temp[-1] < 0.4) or (num_itr > 3)):
        #     iteration = False
    
##%%
    
import joblib

mname = nn_type+'_model.pkl'
joblib.dump(model, dlPath+mname)
rname = nn_type+'_result.pkl'
joblib.dump(result, dlPath+rname)

model = joblib.load(dlPath+mname)
result = joblib.load(dlPath+rname)
    #loss, accuracy = model.evaluate(X_train, y_train)
    #print("loss:", loss, "accuracy:", accuracy)
    
Y_tr_pred = model.predict(X_train[:,5:],steps=1) #model(X_train)
Y_ts_pred = model.predict(x_test[:,5:],steps=1)


#%%
''' ANN
{'activation': 'relu', 'batch_size': 64, 'epochs': 200, 'learning_rate': 0.006085959894190292, 'nodes': 137, 'patience': 7}
MSE:88.50517240074224 - no over, pre-process

{'activation': 'relu', 'batch_size': 256, 'epochs': 100, 'learning_rate': 0.028030941261298955, 'nodes': 195, 'patience': 8}
MSE:117.03654982525599 - sgd, no over, preprocess

{'activation': 'relu', 'batch_size': 64, 'epochs': 200, 'learning_rate': 0.0032604288880638175, 'nodes': 90, 'patience': 17}
MSE:112.29314659590636 - adam, no over, preprocess

{'activation': 'relu', 'batch_size': 256, 'epochs': 100, 'learning_rate': 0.010164644306268199, 'nodes': 146, 'patience': 10}
MSE:113.25926256552083 - over

CNN
{'activation': 'relu', 'batch_size': 128, 'epochs': 100, 'filter_num': 64, 'filter_size': 5, 'learning_rate': 0.002012307357260639, 'nodes': 146, 'patience': 14, 'stride': 2}
MSE:168.16875503559243

{'activation': 'relu', 'batch_size': 64, 'epochs': 100, 'filter_num': 256, 'filter_size': 7, 'learning_rate': 0.0005375400590314959, 'nodes': 57, 'patience': 19, 'stride': 2}
MSE:146.57111096896838 - over

'''
#%% 9. ANALYSIS

#loss function
def plot_history(result):
  hist = pd.DataFrame(result.history)
  hist['epoch'] = result.epoch

  plt.figure(figsize=(8,5))
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['loss'],# marker='.', markersize=2,
           label='Val Loss')
  #plt.plot(hist['epoch'], hist['mse'],
  #         label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  #plt.show()
  plt.savefig('./plot/'+nn_type+'_loss.png')

plot_history(result)

#%% evaluation
X_df = pd.DataFrame(x_test[:,5:])
Y_pred = model.predict(X_df,steps=1)

x = pd.DataFrame(x_test[:,:5])

Y_df = pd.DataFrame(Y_pred, columns=['Pred'])
Y_df['Actual']=y_test
Y_df['Lon']=asarray(x[0])
Y_df['Lat']=asarray(x[1])
Y_df['Day']=asarray(x[3])
Y_df['Hour']=asarray(x[4])

Y_df

res = np.array(np.vstack((x[0].to_numpy(), x[1].to_numpy(), x[3].to_numpy(), x[4].to_numpy(), y_test, Y_pred.flatten())))

r2 = r2_score(y_test, Y_pred)
print('r_squared: '+str(r2))

Y_df.to_csv(dlPath+nn_type+'_result.csv', mode='w')

#%% density plot
from scipy.stats import gaussian_kde

#plt.figure(figsize=(7,7))
#g = sns.jointplot(data=Y_df, x="Actual", y="Pred")
#g.plot_joint(sns.regplot, scatter_kws={"s": 1})

#plt.hist2d(Y_df['Actual'], Y_df['Pred'], bins=(50, 50), cmap=plt.cm.jet)

r2 = r2_score(Y_df['Actual'], Y_df['Pred'])
print('r_squared: '+str(r2))

xy = np.vstack([Y_df['Actual'], Y_df['Pred']])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax = sns.regplot(x="Actual", y="Pred", data=Y_df, scatter_kws={"s": 1}, line_kws={"color":"r", "linewidth":1})
points = ax.scatter(Y_df['Actual'], Y_df['Pred'], c=z, s=1, cmap=plt.cm.jet)
ax.text(.05, .9, '$R^2$={:.3f}'.format(r2), transform=ax.transAxes)
ax.set_title(nn_type+" Evaluation Results")
plt.savefig('./plot/'+nn_type+'_res.png')
#plt.colorbar(points)

#%% pair-wise analysis

X_df = pd.DataFrame(x_test)
err = Y_df['Pred']-Y_df['Actual']

pos = err[err > 15].index
neg = err[err < -15].index

X_df.loc[neg]

test_idx[pos]

pos_err = X_raw[:,test_idx[pos]]
neg_err = X_raw[:,test_idx[neg]]
non_err = np.delete(X_raw, test_idx[pos], axis=1)

title = ['Lon','Lat','Day','Hour','SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES']
idx = [0, 1, 3, 4]+list(range(353, 365))

fig, axes = plt.subplots(4,4, figsize=(14,8), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.scatter(err, X_raw[idx[i],test_idx])
    ax.set_title(title[i])
plt.suptitle(nn_type+' - Analysis')
plt.savefig('./plot/'+nn_type+'_analy.png')

min_idx = list(err).index(min(err))
max_idx = list(err).index(max(err))

print(Y_df['Pred'][min_idx])
print(Y_df['Actual'][max_idx])
#plt.hist(pos_err[3,:])

#X_df.loc[neg].describe().to_csv('./DL/neg_err.csv', mode='w')
#X_df.loc[pos].describe().to_csv('./DL/pos_err.csv', mode='w')
#X.describe().to_csv('./DL/non_err.csv', mode='w')

title = ['Lon','Lat','Day','Hour','SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES']
idx = [0, 1, 3, 4]+list(range(353, 365))

fig, axes = plt.subplots(16,16, figsize=(30,30), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.scatter(X_raw[idx[i%16],test_idx],X_raw[idx[i//16],test_idx], s=0.5, c=err, cmap=plt.cm.jet)
    ax.set_xlabel(title[i%16], fontsize=10)
    ax.set_ylabel(title[i//16], fontsize=10)

plt.suptitle(nn_type+' - Pair Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
plt.savefig('./plot/'+nn_type+'_pair_analy.png')

#%% pair-wise analysis2

X_df = pd.DataFrame(x_test)

title = ['SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES'] #'Lon','Lat','Day','Hour',
idx = list(range(353, 365)) #[0, 1, 3, 4]+

fig, axes = plt.subplots(12,12, figsize=(30,30), constrained_layout=True)
for i, ax in enumerate(axes.flat):
        xy = np.vstack([X_raw[idx[i%12],test_idx],X_raw[idx[i//12],test_idx]])
        z = gaussian_kde(xy)(xy)
        
        ax.scatter(X_raw[idx[i%12],test_idx],X_raw[idx[i//12],test_idx], s=0.5, c=z, cmap=plt.cm.jet)
        ax.set_xlabel(title[i%12], fontsize=10)
        ax.set_ylabel(title[i//12], fontsize=10)

plt.suptitle(nn_type+' - Data Density Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
plt.savefig('./plot/'+nn_type+'_density_pair_analy.png')

#%% pair-wise analysis3

X_df = pd.DataFrame(x_test)

title = ['Lon','Lat','Day','Hour','SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES'] 
idx = [0, 1, 3, 4]+list(range(353, 365)) 

fig, axes = plt.subplots(16,16, figsize=(30,30), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    try:
        im=ax.scatter(X_raw[idx[i%16],test_idx],X_raw[idx[i//16],test_idx], s=0.5, c=Y_df['Pred'], cmap=plt.cm.jet,vmin = 0, vmax =150)
        ax.set_xlabel(title[i%16], fontsize=10)
        ax.set_ylabel(title[i//16], fontsize=10)
    except:
        print(title[i%16]+" and "+title[i//16]+" are singular")

plt.suptitle(nn_type+' - Data Density Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
cbar_ax=fig.add_axes([0.3, 0.06, 0.32, 0.05])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration', orientation='horizontal')   
plt.savefig('./plot/'+nn_type+'_pred_pair_analy.png')


##%% pair-wise analysis4

X_df = pd.DataFrame(x_test)

title = ['Lon','Lat','Day','Hour','SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES'] 
idx = [0, 1, 3, 4]+list(range(353, 365)) 

fig, axes = plt.subplots(16,16, figsize=(30,30), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    try:
        im=ax.scatter(X_raw[idx[i%16],test_idx],X_raw[idx[i//16],test_idx], s=0.5, c=Y_df['Actual'], cmap=plt.cm.jet,vmin = 0, vmax =150)
        ax.set_xlabel(title[i%16], fontsize=10)
        ax.set_ylabel(title[i//16], fontsize=10)
    except:
        print(title[i%16]+" and "+title[i//16]+" are singular")

plt.suptitle(nn_type+' - Data Density Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
cbar_ax=fig.add_axes([0.3, 0.06, 0.32, 0.05])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration', orientation='horizontal')   
plt.savefig('./plot/'+nn_type+'_raw_pair_analy.png')

#%% error analysis

#results load
#Y_df = pd.read_csv(dlPath+nn_type+'_result.csv',sep=",",dtype='unicode') 
#Y_df = Y_df.drop('Unnamed: 0', axis=1)

Y_df['Pred']=pd.to_numeric(Y_df['Pred'])
Y_df['Actual']=pd.to_numeric(Y_df['Actual'])
Y_df['Day']=pd.to_numeric(Y_df['Day'])
Y_df['Hour']=pd.to_numeric(Y_df['Hour'])
Y_df['Lon']=pd.to_numeric(Y_df['Lon'])
Y_df['Lat']=pd.to_numeric(Y_df['Lat'])

X_df = pd.DataFrame(x_test)
err = (Y_df['Pred']-Y_df['Actual'])/Y_df['Actual']*100

pos = err[err > 15].index
neg = err[err < -15].index

X_df.loc[neg], confusion_matrix

test_idx[pos]

pos_err = X_raw[:,test_idx[pos]]
neg_err = X_raw[:,test_idx[neg]]
non_err = np.delete(X_raw, test_idx[pos], axis=1)

title = ['Lon','Lat','Day','Hour','SZA','SAA', 'VZA','VAA','UGRD','VGRD','TMP','RH','DPT','VISIP','HPBL','PRES']
idx = [0, 1, 3, 4]+list(range(353, 365))

fig, axes = plt.subplots(4,4, figsize=(14,8), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.hist(pos_err[idx[i],:], 20)
    ax.set_title(title[i])
plt.suptitle(nn_type+' - Positive Errors')
plt.savefig('./plot/'+nn_type+'_pos.png')

fig, axes = plt.subplots(4,4, figsize=(14,8), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.hist(neg_err[idx[i],:], 20)
    ax.set_title(title[i])
plt.suptitle(nn_type+' - Negative Errors')
plt.savefig('./plot/'+nn_type+'_neg.png')

fig, axes = plt.subplots(4,4, figsize=(14,8), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.hist(non_err[idx[i],:], 20)
    ax.set_title(title[i])
plt.suptitle(nn_type+' - Non Errors')
plt.savefig('./plot/'+nn_type+'_non.png')

#%% explicit comparison

import geopandas as gpd

#library(rworldmap)
#library(ggmap)

#korea_spdf <- rworldmap::getMap(resolution = "high")

kr_file='/home/hsh0514/SHP/korea.shp'
kr_bd=gpd.read_file(kr_file)
kr_bd

for d in range(1,32):
    for h in range(2,8):
        g= Y_df[Y_df['Day']==d]
        g = g[g['Hour']==h]
        print(d, '/',h,':', len(g))

groups = Y_df.groupby(['Lat', 'Lon'])["Pred","Actual"].mean()#.agg({'Pred': ['mean'], 'Actual': ['mean']})

lon = []
lat = []
pred = []
act = []

for (k1, k2), group in Y_df.groupby(['Lon','Lat']):
    #print(k1, k2)
    lon.append(k1)
    lat.append(k2)
    
    av_g = group.mean()
    pred.append(av_g['Pred'])
    act.append(av_g['Actual'])

fig, axes = plt.subplots(1,2, figsize=(14,8))

pred = asarray(pred)
act = asarray(act)
val = [pred, act]
title = ['Predcition', 'Actual Values']
for i, ax in enumerate(axes.flat):
    
    kr_bd.plot(ax = ax, color='white', edgecolor='k', linewidth=0.2)
    im = ax.scatter(lon, lat, c=val[i],cmap='jet', vmin=0, vmax=70, s=4)
    ax.set_title(title[i])
    
val = [Y_df['Pred'], Y_df['Actual']]
for i, ax in enumerate(axes.flat):
    
    kr_bd.plot(ax = ax, color='white', edgecolor='k', linewidth=0.2)
    im = ax.scatter(Y_df['Lon'], Y_df['Lat'], c=val[i], cmap='jet', vmin=0, vmax=70, s=4)
    ax.set_title(title[i])
    
cbar_ax=fig.add_axes([0.3, 0.06, 0.4, 0.05])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration', orientation='horizontal')      
plt.savefig('./plot/'+nn_type+'_comp.png')   
#plt.show()

#%% Feature Importance

#from sklearn.ensemble import ExtraTreesClassifier

#model = ExtraTreesClassifier(n_estimators=10)
#model.fit(X_train, y_train)
#print(model.feature_importances_)

#%% 
print(model.metrics_names) #['loss', 'accuracy']
print(model.evaluate(x_test[:,5:], y_test)) #[6.555384635925293, 0.5]
for name, value in zip(model.metrics_names, model.evaluate(x_test[:,5:], y_test)):
  print(name, value)
  
#y_predict = Y_df['Pred']

#label = labels[1 if y_predict[0][0] > 0.5 else 0]
#confidence = y_predict[0][0] if y_predict[0][0] > 0.5 else 1 - y_predict[0][0]
#print(label, confidence)


#%% pair-wise analysis


Y_df['Pred']=pd.to_numeric(Y_df['Pred'])
Y_df['Actual']=pd.to_numeric(Y_df['Actual'])
Y_df['Day']=pd.to_numeric(Y_df['Day'])
Y_df['Hour']=pd.to_numeric(Y_df['Hour'])
Y_df['Lon']=pd.to_numeric(Y_df['Lon'])
Y_df['Lat']=pd.to_numeric(Y_df['Lat'])

X_df = pd.DataFrame(x_test)
err = (Y_df['Pred']-Y_df['Actual'])#/Y_df['Actual']

pos = err[err > 15].index
neg = err[err < -15].index

X_df.loc[neg]

test_idx[pos]

pos_err = X_raw[:,test_idx[pos]]
neg_err = X_raw[:,test_idx[neg]]
non_err = np.delete(X_raw, test_idx[pos], axis=1)

title = ['UGRD','VGRD','TMP','RH','HPBL']
idx = [357, 358, 359, 360, 363]

min_idx = list(err).index(min(err))
max_idx = list(err).index(max(err))

print(Y_df['Pred'][min_idx])
print(Y_df['Actual'][max_idx])
#plt.hist(pos_err[3,:])

#X_df.loc[neg].describe().to_csv('./DL/neg_err.csv', mode='w')
#X_df.loc[pos].describe().to_csv('./DL/pos_err.csv', mode='w')
#X.describe().to_csv('./DL/non_err.csv', mode='w')

fig, axes = plt.subplots(5,5, figsize=(10,10), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    im = ax.scatter(X_raw[idx[i%5],test_idx],X_raw[idx[i//5],test_idx], s=0.5, c=err, vmin=-50, vmax=50, cmap=plt.cm.jet)
    ax.set_xlabel(title[i%5], fontsize=10)
    ax.set_ylabel(title[i//5], fontsize=10)

plt.suptitle(nn_type+' - Pair Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
cbar_ax=fig.add_axes([0.3, 0.03, 0.32, 0.02])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration Error Rate', orientation='horizontal')  
plt.savefig('./plot/'+nn_type+'_pair_analy.png')

#%% pair-wise analysis2

X_df = pd.DataFrame(x_test)

title = ['UGRD','VGRD','TMP','RH','HPBL']
idx = [357, 358, 359, 360, 363]

fig, axes = plt.subplots(5,5, figsize=(10,10), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    try:
        xy = np.vstack([X_raw[idx[i%5],test_idx],X_raw[idx[i//5],test_idx]])
        z = gaussian_kde(xy)(xy)
        
        ax.scatter(X_raw[idx[i%5],test_idx],X_raw[idx[i//5],test_idx], s=0.5, c=z, cmap=plt.cm.jet)
        ax.set_xlabel(title[i%5], fontsize=10)
        ax.set_ylabel(title[i//5], fontsize=10)
    except: 
        print(title[i%5]+" and "+title[i//5]+" are singular")


plt.suptitle(nn_type+' - Data Density Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
plt.savefig('./plot/'+nn_type+'_density_pair_analy.png')

#%% pair-wise analysis3

X_df = pd.DataFrame(x_test)
title = ['UGRD','VGRD','TMP','RH','HPBL']
idx = [357, 358, 359, 360, 363] 

fig, axes = plt.subplots(5,5, figsize=(10,10), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    try:
        im=ax.scatter(X_raw[idx[i%5],test_idx],X_raw[idx[i//5],test_idx], s=0.5, c=Y_df['Pred'], cmap=plt.cm.jet,vmin = 0, vmax =150)
        ax.set_xlabel(title[i%5], fontsize=10)
        ax.set_ylabel(title[i//5], fontsize=10)
    except:
        print(title[i%5]+" and "+title[i//5]+" are singular")

plt.suptitle(nn_type+' - Pred Concentration Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
cbar_ax=fig.add_axes([0.3, 0.03, 0.32, 0.02])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration', orientation='horizontal')   
plt.savefig('./plot/'+nn_type+'_pred_pair_analy.png')


##%% pair-wise analysis4

fig, axes = plt.subplots(5,5, figsize=(10,10), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    try:
        im=ax.scatter(X_raw[idx[i%5],test_idx],X_raw[idx[i//5],test_idx], s=0.5, c=Y_df['Actual'], cmap=plt.cm.jet,vmin = 0, vmax =150)
        ax.set_xlabel(title[i%5], fontsize=10)
        ax.set_ylabel(title[i//5], fontsize=10)
    except:
        print(title[i%5]+" and "+title[i//5]+" are singular")

plt.suptitle(nn_type+' - Raw Concentration Analysis')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
cbar_ax=fig.add_axes([0.3, 0.03, 0.32, 0.02])  # left, bottom, width, height

fig.colorbar(im,cax=cbar_ax,label='PM2.5 concentration', orientation='horizontal')   
plt.savefig('./plot/'+nn_type+'_raw_pair_analy.png')

