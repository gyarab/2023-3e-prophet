#Imports for caluclating indicators
import csv # lib for csv files access
from tqdm import tqdm # loop progress bar in terminal
import os # file paths
import pandas as pd
import ta

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







# adds anotation off technical indicators and target values and their differences
def AddTechnicalIndicators(csv_file, output_file):
    

    # Add target_value and target_value_difference columns
    df['target_value'] = df['close'].shift(-1)  # Next close value
    df['target_value_difference'] = df['target_value'] - df['close']  # Difference between next close and current close
    # Drop the 'timestamp' column
    df.drop('timestamp', axis=1, inplace=True)  
    # Reorder columns
    columns_order = ['date', 'target_value', 'target_value_difference', 'open', 'high', 'low', 'close', 'volume']
    columns_order.extend([col for col in df.columns if col not in columns_order])
    df = df[columns_order]
   
   
    # Add technical indicators
    df['ema_14'] = ta.trend.EMAIndicator(close=df['close'], window=14).ema_indicator()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_upper'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_lower'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['ichimoku_a'] = ta.trend.IchimokuIndicator(high=df['high'], low=df['low']).ichimoku_a()
    df['ichimoku_b'] = ta.trend.IchimokuIndicator(high=df['high'], low=df['low']).ichimoku_b()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(close=df['close'], high=df['high'], low=df['low']).williams_r()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    
   
    
    


if __name__ == '__main__':
    # print(get_btc_price())
    #print("CoinEx Account Balance:")
    #print(get_balance())
    
    for candle in get_last_100_btc_price():
        timestamp, open_, high, low, close, volume = candle
        print(f'Timestamp: {timestamp}, Close Price: {close}')
     




last_100_edited_prices = get_last_100_btc_price
    