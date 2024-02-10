#Imports for caluclating indicators
import csv # lib for csv files access
from tqdm import tqdm # loop progress bar in terminal
import os # file paths
import pandas as pd
import ta
import numpy as np
#Getting values
import ccxt
import tqdm

APIKEY = '01566309D7D54F6B83CD7BD57090B485'
SECRETKEY = 'FF1A020B10EC704972C034475F2BBA140F814F87B401F37B'

EXCHANGE = ccxt.coinex({
        'apiKey': APIKEY,
        'secret': SECRETKEY,
    })

def get_btc_price():
    symbol = 'BTC/USDT'
    # Fetch ticker information for BTC/USDT pair
    ticker = EXCHANGE.fetch_ticker(symbol)

    # Extract and print the last price
    last_price = ticker['last']
    
    return last_price

def get_last_100_btc_price():
    symbol = 'BTC/USDT'
    
    # Fetch historical OHLCV data with 1-minute timeframe
    #OHLCV stands for: open, high, low, close, volume
    ohlcv = EXCHANGE.fetch_ohlcv(symbol, '1m') # Use '1m' for 1-minute timeframe

    # last 100 minutes as ?list?
    last_100_prices = ohlcv[-100:]    
    return last_100_prices

#returns account balance as dictoniary
def get_balance():
    # Fetch your account balance
    balance = EXCHANGE.fetch_balance()
    return balance

def get_markets():
    markets = EXCHANGE.load_markets()
    return list(markets.keys())

    
    
def Create_price_arr():
    

    # Create an empty list to store rows of the array
    array_data = []

    for candle in get_last_100_btc_price():
        timestamp, open_, high, low, close, volume = candle

        # Convert values to Pandas Series
        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        volume_series = pd.Series(volume)

        # Calculate technical indicators
        ema_14 = ta.trend.EMAIndicator(close=close_series, window=14).ema_indicator()
        rsi_14 = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
        macd = ta.trend.MACD(close=close_series).macd()
        bollinger_upper = ta.volatility.BollingerBands(close=close_series).bollinger_hband()
        bollinger_lower = ta.volatility.BollingerBands(close=close_series).bollinger_lband()

        # Check if there are enough data points for ATR calculation
        if len(close_series) >= 14:
            atr = ta.volatility.AverageTrueRange(high=high_series, low=low_series, close=close_series).average_true_range()
        else:
            atr = None  # Set to None if not enough data
        
        # Check if there are enough data points for ADXIndicator calculation
        if len(close_series) >= 2:
            adx = ta.trend.ADXIndicator(high=high_series, low=low_series, close=close_series).adx()
        else:
            adx = None  # Set to None if not enough data

        ichimoku_a = ta.trend.IchimokuIndicator(high=high_series, low=low_series).ichimoku_a()
        ichimoku_b = ta.trend.IchimokuIndicator(high=high_series, low=low_series).ichimoku_b()
        obv = ta.volume.OnBalanceVolumeIndicator(close=close_series, volume=volume_series).on_balance_volume()
        williams_r = ta.momentum.WilliamsRIndicator(close=close_series, high=high_series, low=low_series).williams_r()
        

        # Create a row with all values
        #row = [timestamp, open_, high, low, close, volume, ema_14, rsi_14, macd, bollinger_upper, bollinger_lower, atr, ichimoku_a, ichimoku_b, obv, williams_r, adx]
        row = [close]
        
        
        
        
        
        # Append the row to the array_data list
        array_data.append(row)


    
    return array_data
    

if __name__ == '__main__':
    # print(get_btc_price())
    #print("CoinEx Account Balance:")
    #print(get_balance())
    

    last_prices = Create_price_arr()
    print('Array output:')
    for row in last_prices:
        print(row)


    for candle in get_last_100_btc_price():
        timestamp, open_, high, low, close, volume = candle
        #print(f'Timestamp: {timestamp}, Close Price: {close}')
        
   

