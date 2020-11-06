#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:58:59 2020

@author: ryanfinegan
"""

#Libraries
import pandas as pd
import random
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.ensemble import AdaBoostRegressor (used this but rfr worked better)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

#Macro possible drivers of Real GDP
def macro(end=dt.datetime.now(),start = '2000-01-01'):
    # housing starts, cpi, 10 yr treasury, federal funds rate, 3 month libor, unemployment rate, non-farm, cboe vol index
    symbols_list = 'HOUST','CPIAUCSL','PAYEMS','VIXCLS','MANEMP','GDPC1'
    #empty symbols list to add data to after the for loop
    symbols = []
    #going to fred to get all the data with a try clause
    for ticker in symbols_list: 
        try:
            r = web.DataReader(ticker,'fred',start,end)
            r[f'{ticker}'] = r[f'{ticker}']
            r1 = r[[f'{ticker}']]
            symbols.append(r1)
        except:
            msg = 'Failed to read symbol: {0!r}, replacing with NaN.'
    #concatenating all values into the same dataframe
    df = pd.concat(symbols, sort=False, axis=1)
    #fill any values due to quarterly data cleaning
    df = df.fillna(method='ffill')
    #sample by quarter
    df = df.resample('Q',convention='start').asfreq()
    #shifting the target to predict the next quarter of gdp
    df['GDPC1'] = df['GDPC1'].shift(-1)
    #drop any NaN values 
    df = df.dropna(inplace=False)
    return df


def gdp_predict_rfr(df):
    #splitting the data set into train, test, and validation
    train,test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state = 222)
    train,val = sklearn.model_selection.train_test_split(train, test_size=0.2, random_state = 222)
    #setting up the target variable (real GDP)
    trainy = train['GDPC1']
    del train['GDPC1']
    valy = val['GDPC1']
    del val['GDPC1']
    testy = test['GDPC1']
    del test['GDPC1']
    #RandomForestRegressor
    model = RandomForestRegressor(random_state=0, n_estimators=100)
    #fitting the training features and targets to the model 
    model.fit(train, trainy)
    #getting the accuracy score from the validation data set
    score = model.score(val,valy)
    #getting the accuracy score from the testing dataset
    modprob = model.score(test,testy)
    #prediction
    symbols_list = 'HOUST','CPIAUCSL','PAYEMS','VIXCLS','MANEMP'
    #empty symbols list for the real-time features for prediction
    symbols = []
    for ticker in symbols_list: 
        try:
            r = web.DataReader(ticker,'fred','2020-01-01',dt.datetime.now())
            r[f'{ticker}'] = r[f'{ticker}']
            r1 = r[[f'{ticker}']]
            symbols.append(r1)
        except:
            msg = 'Failed to read symbol: {0!r}, replacing with NaN.'
    #combining the prediction features into one dataframe
    macro = pd.concat(symbols, sort=False, axis=1)
    #filling NaN values since different macro events are announced on different days
    macro = macro.fillna(method='ffill')
    #resampling the dataset from daily to quarterly 
    macro = macro.resample('Q',convention='start').asfreq()
    #dropping any NaN values
    macro = macro.dropna()
    #prediction features for real GDP forecasting
    p1,p2,p3,p4,p5 = macro.iloc[-1][0],macro.iloc[-1][1],macro.iloc[-1][2],macro.iloc[-1][3],macro.iloc[-1][4]
    #final real GDP prediction using sklearn regressor 
    prediction = model.predict([[p1,p2,p3,p4,p5]])
    #printing the accuracy scores and prediction point 
    print1 = (f'Model Validation Data Score: {score}')
    print2 = (f'Model Test Data Score: {modprob}')
    print3 = (f'Next US Real GDP: ${prediction[0]}')
    return print1, print2, print3,macro.tail(5),df

if __name__ == '__main__':
    gdp_predict_rfr(df = macro(end=dt.datetime.now(),start = '1980-01-01'))
    
