# This script is for simulating trading on live data (forward testing)
import time
from copy import deepcopy as dc
import numpy as np
import torch
import best_brain as bb
import binance_data_fetcher as bdf
from data_manager import LoaderOHLCV
import trader
import json_data_handler
is_trading = False

symbol = 'BTCUSDT'
look_back = 9
# trading data
td ={
    "USD_balance": 0,
    "BTC_balance": 69,
    "long_count": 0,
    "short_count": 0,
    "hold_count": 0,
    "bad_trade_count": 0,
    "good_trade_count": 0,
    "total_profit": 0,
    "total_loss": 0,
    "leverage": 1,
    "comission_rate": 0
}
bb.model = bb.load_data_model(bb.model, bb.model_path)
bb.model.to(bb.device)

def initialize_start_balance(start_usd_balance = 10000):
    global td
    td["USD_balance"] = start_usd_balance
    td["BTC_balance"] = 0

def start_trading():
    global is_trading 
    is_trading = True
def stop_trading():
    global is_trading
    is_trading = False
def train_loop():
    global td
    last_trade = 'hold'
    last_balance = td["USD_balance"]
    DataManager =  LoaderOHLCV(look_back,['Close'], mode= 3) # !!! HARDCODED
    while is_trading == True:
        # Gets raw data from data_fetcher
        raw_data = bdf.get_live_minute_datapoints(symbol, lookback = look_back)
        # Get the last value - the most actual BTC price 
        current_btc_price = float(raw_data['Close'].iloc[-1])
        # Prepares data
        one_sequence_tensor = DataManager.prepare_live_data(raw_data)
        # Models makes prediction
        prediction = bb.make_one_prediction(one_sequence_tensor)
        # Simulates one trade
        td["USD_balance"], td["BTC_balance"], last_trade = trader.make_one_trade(prediction,td["USD_balance"],td["BTC_balance"],current_btc_price,td["comission_rate"], last_trade, td["leverage"])        
        # Calculates how much usd would he have if he closed trade
        after_close_usd_balance, _ = trader.close_trade(td["USD_balance"], td["BTC_balance"], current_btc_price, td["comission_rate"])
        # Updates stats
        calculate_stats(last_trade, after_close_usd_balance, last_balance)
        save_all_trading_data()
        last_balance = after_close_usd_balance       
        
        # Waits minute
        time.sleep(60)
def calculate_stats(last_trade, after_close_usd_balance, last_balance):
    global td
    if last_trade == 'long':
        td["long_count"] += 1
    elif last_trade == 'short':
        td["short_count"] += 1
    else:
        td["hold_count"] +=1
    if after_close_usd_balance > last_balance:
        td["good_trade_count"] +=1
        td["total_profit"] += (after_close_usd_balance - last_balance)        
    else:
        td["bad_trade_count"] +=1
        td["total_loss"] += (last_balance - after_close_usd_balance)
    pass
def save_all_trading_data():
    for key, value in td.items():
        # Call the function with unpacked kwargs
        json_data_handler.update_trading_data(key=key,value= value)
if __name__ == '__main__':
    initialize_start_balance()
    start_trading()
    train_loop()