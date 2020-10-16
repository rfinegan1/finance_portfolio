#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:21:09 2020

@author: ryanfinegan
"""

# main libraries and yahoo finance api
import random
import yfinance as yf
import numpy as np

# machine learning libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Concatenate

# warnings library to ignore all warnings with code for cleaner viewing
import warnings
warnings.filterwarnings('ignore')

# functions for dataframe creation and machine learning neural network to predict intraday prices
#crypto dataset creation
def cryptoDataset(tickers, period, interval):
    df = yf.download(tickers = tickers.upper(),period = period,interval = interval)[['Adj Close']]
    df['target'] = df['Adj Close'].shift(-1)
    df['sm3'] = df['Adj Close'].rolling(window=3).mean()
    df['sm15'] = df['Adj Close'].rolling(window=15).mean()
    df = df.drop('Adj Close',axis=1)
    return df

#rsi for intraday study (will be added to the crypto dataset)
#was removed from the features after increasing the validation mean absolute percentage error
def relative_strength_idx(df, n=14):
    close = df['target']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

#neural network
def stock_predictor_model(x_train):
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1],
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
    return model

#dataset for final prediction
def predDataset(tickers, period, interval):
    df = yf.download(tickers = tickers.upper(),period = period,interval = interval)[['Adj Close']]
    df['sm3'] = df['Adj Close'].rolling(window=3).mean()
    df['sm9'] = df['Adj Close'].rolling(window=9).mean()
    df['sm15'] = df['Adj Close'].rolling(window=15).mean()
    df['sm35'] = df['Adj Close'].rolling(window=35).mean()
    df['sm75'] = df['Adj Close'].rolling(window=75).mean()
    return df

# main function for predicting the next movement 
def predict(target_security='aapl'):
    df = cryptoDataset(tickers = target_security,period = '1mo',interval = '2m')
    rsi = relative_strength_idx(df, n=14).dropna()
    df = df.dropna()
    features = df.loc[:,df.columns!='target']
    X = df[features.columns]
    Y = df[['target']]
    random.seed(100)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=50,test_size=0.2)
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,random_state=50,test_size=0.5)
    batch_size = 32

    model = stock_predictor_model(X_train)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    history = model.fit(X_train, Y_train, 
                        validation_data=[X_val, Y_val],
                        batch_size=batch_size,
                        epochs=2,
                        verbose=1)
    
    
    pred = predDataset(tickers = target_security,period = '1mo',interval = '2m')
    pred_df = pred[features.columns]
    pred_features = pred_df.iloc[-1]
    prediction = model.predict(np.array([pred_features]))
    print(f'The predicted stock price for {target_security.upper()} in the next two minutes is ${float(prediction[0])}.')
    if float(prediction[0])>float(pred['Adj Close'].iloc[-1:].values):
        print('Long: ', float(prediction[0]), '>',float(pred['Adj Close'].iloc[-1:].values),'\nPCT DIFF: ',(float(prediction[0]) - float(pred['Adj Close'].iloc[-1:].values))/float(pred['Adj Close'].iloc[-1:].values)*100,'%')
    else:
        print('Short: ', float(prediction[0]), '<',float(pred['Adj Close'].iloc[-1:].values),'\nPCT DIFF: ',(float(prediction[0]) - float(pred['Adj Close'].iloc[-1:].values))/float(pred['Adj Close'].iloc[-1:].values)*100,'%')
        
if __name__ == '__main__':
    predict('aapl')