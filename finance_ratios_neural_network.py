#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:15:44 2020

@author: ryanfinegan
"""

# main libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# machine learning libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Concatenate
from keras.optimizers import RMSprop

# warnings libraries
import warnings
warnings.filterwarnings('ignore')

# api for financial information and valuation ratios
import pandas_datareader.data as web

# make dataframe out of a csv file
df1 = pd.read_csv('2018_Financial_Data.csv', index_col=0).drop(columns = ['Sector'])
df2 = pd.read_csv('2017_Financial_Data.csv', index_col=0).drop(columns = ['Sector'])
df3 = pd.read_csv('2016_Financial_Data.csv', index_col=0).drop(columns = ['Sector'])
df4 = pd.read_csv('2015_Financial_Data.csv', index_col=0).drop(columns = ['Sector'])
df5 = pd.read_csv('2014_Financial_Data.csv', index_col=0).drop(columns = ['Sector'])

# function to create the yearly dataframe of fundamental year over year rates
def fundamental_rates(dataframe, year):
    '''takes dataframe of a certain year and the following year as inputs. 
    Returns a dataframe with the fundamental growth rates of each stock with
    the returns from the next year.'''
    df = (dataframe.columns.str.contains('Growth') | dataframe.columns.str.contains('Class') | dataframe.columns.str.contains('VAR'))
    columns = dataframe.columns[df]
    df = dataframe[columns]
    df['target'] = df[year+' PRICE VAR [%]']
    df = df.drop(['Class',year+' PRICE VAR [%]'], axis = 1)
    return df.dropna()

# function to create teh yearly dataframe of fundamental year over year rates with binary stock performance (positive / negative returns)
def fundamental_binary(dataframe, year):
    '''takes dataframe of a certain year and the following year as inputs.
    Returns a dataframe with the fundamental growth rates of each stock with 
    the returns from the next year.'''
    df = (dataframe.columns.str.contains('Growth') | dataframe.columns.str.contains('Class') | dataframe.columns.str.contains('VAR'))
    columns = dataframe.columns[df]
    df = dataframe[columns]
    df['target'] = df['Class']
    df = df.drop(['Class',year+' PRICE VAR [%]'], axis = 1)
    return df.dropna()

# random forest regressor to find important features
def feature_importance_regressor(dataframe):
    X = dataframe.loc[:,dataframe.columns!='target']
    Y = dataframe['target']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=50,test_size=0.2)
    rfr = RandomForestRegressor(random_state=50,oob_score=True,max_features='sqrt')
    rfr.fit(X_train,Y_train)
    y_rfr_pred = rfr.predict(X_test)
    print('Train r squared score:',r2_score(Y_train,rfr.predict(X_train)))
    print('Test r squared score:',r2_score(Y_test,rfr.predict(X_test)))
    features = X_train.columns
    importances = rfr.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10,10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),importances[indices],color='r',align='center')
    plt.yticks(range(len(indices)),[features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    new_features = features[indices][-6:]
    return new_features

# fundamental yearly percentage performance dataframes
yr_2018 = fundamental_rates(df1,'2019')
yr_2017 = fundamental_rates(df2,'2018')
yr_2016 = fundamental_rates(df3,'2017')
yr_2015 = fundamental_rates(df4,'2016')
yr_2014 = fundamental_rates(df5,'2015')

# fundamental yearly binary performance dataframes
bin_2018 = fundamental_binary(df1,'2019')
bin_2017 = fundamental_binary(df2,'2018')
bin_2016 = fundamental_binary(df3,'2017')
bin_2015 = fundamental_binary(df4,'2016')
bin_2014 = fundamental_binary(df5,'2015')

# new features from Random Forest Regressor
new_features_2018 = feature_importance_regressor(yr_2018)
new_features_2017 = feature_importance_regressor(yr_2017)
new_features_2016 = feature_importance_regressor(yr_2016)
new_features_2015 = feature_importance_regressor(yr_2015)
new_features_2014 = feature_importance_regressor(yr_2014)

# neural network to predict next years stock performance using yearly financial growth rates
def fundamental_stock_pct_nn(dataframe,new_features):
    X = dataframe[new_features]
    Y = dataframe[['target']]
    random.seed(100)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=50,test_size=0.2)    
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,random_state=50,test_size=0.5)
    batch_size = 32
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1],
                    activation=tf.nn.leaky_relu,
                    kernel_initializer='he_normal'))
    model.add(Dense(60, input_dim=100,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer='he_normal'))
    model.add(Dense(30, input_dim=60,
                activation=tf.nn.leaky_relu,
                kernel_initializer='he_normal'))
    model.add(Dense(1, activation=tf.nn.leaky_relu,
                    kernel_initializer='he_normal'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mape'])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    history = model.fit(X_train, Y_train, 
                    validation_data=[X_val, Y_val],
                    batch_size=batch_size,
                    epochs=2,
                    verbose=1)
    return model

# feature importance using a random forest classifier 
def feature_importance_classifier(dataframe):
    X = dataframe.loc[:,dataframe.columns!='target']
    Y = dataframe['target']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=50,test_size=0.2)
    rfc = RandomForestClassifier(random_state=50,oob_score=True,max_features='sqrt')
    rfc.fit(X_train,Y_train)
    y_rfc_pred = rfc.predict(X_test)
    print('Train r squared score:',r2_score(Y_train,rfc.predict(X_train)))
    print('Test r squared score:',r2_score(Y_test,rfc.predict(X_test)))
    features = X_train.columns
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(2,2))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),importances[indices],color='r',align='center')
    plt.yticks(range(len(indices)),[features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    new_features = features[indices][-6:]
    return new_features 

# new features from Random Forest Classifier
bin_new_features_2018 = feature_importance_classifier(bin_2018)
bin_new_features_2017 = feature_importance_classifier(bin_2017)
bin_new_features_2016 = feature_importance_classifier(bin_2016)
bin_new_features_2015 = feature_importance_classifier(bin_2015)
bin_new_features_2014 = feature_importance_classifier(bin_2014)

# binary neural network to predict next years stock performance (up or down) using yearly financial growth rates
def binary_fundamental_stock_nn(dataframe,new_features):
    X = dataframe[new_features]
    Y = dataframe[['target']]
    random.seed(100)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=50,test_size=0.2)    
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,random_state=50,test_size=0.5)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(Y_train)
    y_test = np.array(Y_test)
    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10,activation = tf.nn.leaky_relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    # Compile with optimizer, loss fxn, and metric (accuracy is good for classification models)
    simple_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
    #preprocessing 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Fitting to the training set
    simple_model.fit(X_train_scaled,y_train,batch_size=32,epochs=3,validation_data=(X_test,y_test))
    
#neural networks for guessing the percentage 
model = fundamental_stock_pct_nn(yr_2018,new_features_2018)
model1 = fundamental_stock_pct_nn(yr_2018,new_features_2017)
model2 = fundamental_stock_pct_nn(yr_2018,new_features_2016)
model3 = fundamental_stock_pct_nn(yr_2018,new_features_2015)
model4 = fundamental_stock_pct_nn(yr_2018,new_features_2014)

#binary neural networks for guessing if a stock will have positive or negative returns next year
bin_model = binary_fundamental_stock_nn(bin_2018,bin_new_features_2018)
bin_model1 = binary_fundamental_stock_nn(bin_2017,bin_new_features_2017)
bin_model2 = binary_fundamental_stock_nn(bin_2016,bin_new_features_2016)
bin_model3 = binary_fundamental_stock_nn(bin_2015,bin_new_features_2015)
bin_model4 = binary_fundamental_stock_nn(bin_2014,bin_new_features_2014)
    
    
