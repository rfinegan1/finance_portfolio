#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:27:46 2020

@author: ryanfinegan
"""

import pandas as pd
import requests

def up():
    # wall street journal daily stock upgrades and downgrades
    url = 'https://www.wsj.com/market-data/stocks/upgradesdowngrades'
    headers={'User-Agent':'Mozilla/5.0'}
    #submitting a url request
    r = requests.get(url, headers=headers)
    #finding only the stock upgrades
    u = r.text.split('Upgraded')
    #only want around 20 stocks
    upgrades = u[1:25]
    #selecting all the tickers
    upgrades = str(upgrades).rsplit('ticker":')[1:]
    #creating the variables I want updated on everyday
    tickers = []
    price_target = []
    rating_change = []
    
    #for loops to find the ticker, price target, and rating change of each stock
    for i in upgrades:
        names = i.split(',')[0]
        tickers.append(names)
    for i in str(upgrades).rsplit('priceTarget":')[1:]:
        target = i.split('},')[0]
        price_target.append(target)
    for i in str(upgrades).rsplit('ratingsChange":')[1:]:
        rating = i.split(',')[0]
        rating_change.append(rating)
    
    #cleaning the dataframe
    df = pd.DataFrame({'Price Target':price_target, 'Rating Update': rating_change}, index = [tickers])
    #changing the index name from 0 to Ticker
    df.index.names = ['Ticker']
    #removing the index
    df = df.reset_index()
    #replacing the " " in each cell
    df['Ticker'] = df['Ticker'].str.replace(r"[\"]", '')
    df['Price Target'] = df['Price Target'].str.replace(r"[\"]", '')
    df['Rating Update'] = df['Rating Update'].str.replace(r"[\"]", '')
    return df

if __name__ == '__main__':
    up()
