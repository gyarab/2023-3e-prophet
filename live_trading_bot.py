# This script is for simulating trading on live data (forward testing)
import time
from copy import deepcopy as dc
import numpy as np
import torch
import best_brain as bb
import binance_data_fetcher as bdf
from data_manager import LoaderOHLCV
import trader
is_trading = False

symbol = 'BTCUSDT'
usd_balance = 10000
btc_balance = 0
leverage = 1
comission_rate = 0
look_back = 9

bb.model = bb.load_data_model(bb.model, bb.model_path)
bb.model.to(bb.device)

def initialize_start_balance(usd_balance = 10000):
    btc_balance = 0
    return usd_balance, btc_balance
def start_trading():
    global is_trading 
    is_trading = True
def stop_trading():
    global is_trading
    is_trading = False
def train_loop():
    last_trade = 'hold'
    while is_trading == True:
        global usd_balance
        global btc_balance
        # Gets raw data from data_fetcher
        raw_data = bdf.get_live_minute_datapoints(symbol, lookback = look_back)
        # Get the last value - the most actual BTC price 
        current_btc_price = float(raw_data['Close'].iloc[-1])
        
        DataManager =  LoaderOHLCV(look_back,['Close'], 3) # !!! HARDCODED
        # Prepares data for prediction
        prepared_data = DataManager.prepare_dataframe_for_lstm3(raw_data, train= False)
        prepared_data_as_np = prepared_data.to_numpy()
        # Converts the data to correct chronological order
        prepared_data_as_np = dc(np.flip(prepared_data_as_np, axis= 1))
        # Tranforms sequence to correct form
        one_sequence = prepared_data_as_np.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        # models makes prediction
        prediction = bb.make_one_prediction(one_sequence_tensor)
        # simulates one trade
        usd_balance, btc_balance, last_trade = trader.make_one_trade(prediction,usd_balance,btc_balance,current_btc_price, comission_rate, last_trade, leverage)        
        # waits minute
        time.sleep(60)
#usd_balance, btc_balance = initialize_start_balance()
start_trading()
train_loop()