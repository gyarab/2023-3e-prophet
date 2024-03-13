import pandas as pd
from binance.client import Client
import datetime
import os

# YOUR API KEYS HERE
# asi by to chtelo fixnout...
api_key = "lol"    #Enter your own API-key here
api_secret = "xd" #Enter your own API-secret here

bclient = Client(api_key=api_key, api_secret=api_secret)

start_date = datetime.datetime.strptime('1 Jan 2024', '%d %b %Y')
today = datetime.datetime.today()

def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    return input_file_path

def binanceBarExtractor(symbol):
    print('downloading historical data from Binance...')
    filename = get_absolute_path('MinuteBars.csv')

    klines = bclient.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:%M:%S"), 1000)
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 
                                           'volume', 'close_time', 'quote_av', 'trades'
                                           , 'tb_base_av', 'tb_quote_av', 'ignore' ])
    # converts time stamp to date
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.drop('timestamp', axis=1)
    data.set_index('Date', inplace=True)
    # removes no important columns
    drop_column_names = ['close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
    for column_name in drop_column_names:
        data = data.drop(column_name, axis= 1)
    # rename columns to start with a capital 
    rename_column_name = ['open', 'high', 'low', 'close', 'volume']
    for column_name in rename_column_name:
        # capitalize makes first letter capital
        data[column_name.capitalize()] = data[column_name]
        data = data.drop(column_name, axis= 1)
    # saves the csv file
    data.to_csv(filename)
    print('finished!')


if __name__ == '__main__':
    
    binanceBarExtractor('BTCUSDT')