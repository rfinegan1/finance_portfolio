#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:18:02 2020

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
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

#XMPT is a municipal bond etf that is mainly made up of Nuveen Muni funds

#Main ETF function to get the neccessary long term data
def etf(ticker='xmpt',end=dt.datetime.now(),start = '2000-01-01'): 
    etf = web.DataReader(ticker,'yahoo',start,end)[['Close','High','Low']]
    etf['Target'] = etf['Close'].shift(-1)
    etf['sm15'] = etf['Close'].rolling(window=15).mean()
    etf = etf.drop('Close',axis=1)
    etf = etf.dropna()
    return etf

#Macro possible drivers of fixed income performance
def macro(end=dt.datetime.now(),start = '2000-01-01'):
    symbols_list = 'CPIAUCSL','DGS10','FEDFUNDS','USD3MTD156N','TEDRATE','UNRATE','PAYEMS','IC4WSA','VIXCLS','BAMLCC0A0CMTRIV','BAMLHYH0A0HYM2TRIV'
    symbols = []
    for ticker in symbols_list: 
        try:
            r = web.DataReader(ticker,'fred',start,end)
            r[f'{ticker}'] = r[f'{ticker}']
            r1 = r[[f'{ticker}']]
            symbols.append(r1)
        except:
            msg = 'Failed to read symbol: {0!r}, replacing with NaN.'

    df = pd.concat(symbols, sort=False, axis=1)
    df = df.fillna(method='ffill')
    df = df.dropna()
    return df

#Function to combine the ETF and macro driver dataframes
def combine(etf,df):
    data = pd.concat([etf,df],axis=1)
    data = data.dropna()
    return data

def main(df,ticker):
    #resampling data based on quarterly
    df = df.resample('M',convention='start').asfreq()
    df = df.dropna(inplace=False)
    #splitting the data set into train, test, and validation
    train,test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state = 222)
    train,val = sklearn.model_selection.train_test_split(train, test_size=0.2, random_state = 222)
    trainy = train['Target']
    del train['Target']
    valy = val['Target']
    del val['Target']
    testy = test['Target']
    del test['Target']
    #AdaBoost Regressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.datasets import make_regression
    model = AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(train, trainy)
    score = model.score(val,valy)
    modprob = model.score(test,testy)
    #prediction
    pred = web.DataReader('ILTB','yahoo','2000-01-01',dt.datetime.now())[['High','Low','Close']]
    pred['sm15'] = pred['Close'].rolling(window=15).mean()
    pred = pred.drop('Close',axis=1)
    pred = pred.resample('M').first()
    pred = pred.resample('M',convention='start').asfreq()
    pred = pred.dropna(inplace=False)
    #High
    p1 = pred.iloc[-1][0]
    #Low
    p2 = pred.iloc[-1][1]
    #15 Day Moving Average
    p3 = pred.iloc[-1][2]
    #3 month LIBOR, 3 month libor to treasury spread, unemployment rate, nonfarm employees
    #4 week moving average of initial claims, VIX, BAML US Corporate Index, BAML US HY Index
    symbols_list = 'CPIAUCSL','DGS10','FEDFUNDS','USD3MTD156N','TEDRATE','UNRATE','PAYEMS','IC4WSA','VIXCLS','BAMLCC0A0CMTRIV','BAMLHYH0A0HYM2TRIV'
    symbols = []
    for ticker in symbols_list: 
        try:
            r = web.DataReader(ticker,'fred','2020-01-01',dt.datetime.now())
            r[f'{ticker}'] = r[f'{ticker}']
            r1 = r[[f'{ticker}']]
            symbols.append(r1)
        except:
            msg = 'Failed to read symbol: {0!r}, replacing with NaN.'

    macro = pd.concat(symbols, sort=False, axis=1)
    macro = macro.fillna(method='ffill')
    macro = macro.dropna()
    p4,p5,p6,p7 = macro.iloc[-1][0],macro.iloc[-1][1],macro.iloc[-1][2],macro.iloc[-1][3]
    p8,p9,p10,p11 = macro.iloc[-1][4],macro.iloc[-1][5],macro.iloc[-1][6],macro.iloc[-1][7]
    p12,p13,p14 = macro.iloc[-1][8],macro.iloc[-1][9],macro.iloc[-1][10]
    prediction = model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14]])
    print1 = (f'Model Validation Data Score: {score}')
    print2 = (f'Model Test Data Score: {modprob}')
    print3 = (f'Next Close Price: ${prediction[0]}')
    return print1, print2, print3,df

ticker='XMPT'
main_df = combine(etf = etf(ticker,end=dt.datetime.now(),start = '2000-01-01'), 
        df = macro(end=dt.datetime.now(),start = '2000-01-01')) 
df = main_df
if __name__ == '__main__':
    main(df, ticker)
    
main(df, ticker)