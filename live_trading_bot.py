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
usd_balance = 0
btc_balance = 0
leverage = 1
comission_rate = 0
look_back = 9

long_count = 0
short_count = 0
hold_count = 0
bad_trade_count = 0
good_trade_count = 0
good_trade_profit = 0
bad_trade_loss = 0
bb.model = bb.load_data_model(bb.model, bb.model_path)
bb.model.to(bb.device)

def initialize_start_balance(start_usd_balance = 10000):
    global usd_balance, btc_balance
    usd_balance = start_usd_balance
    btc_balance = 0
    return usd_balance, btc_balance

def start_trading():
    global is_trading 
    is_trading = True
def stop_trading():
    global is_trading
    is_trading = False
def train_loop():
    global usd_balance, btc_balance
    last_trade = 'hold'
    last_balance = usd_balance
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
        usd_balance, btc_balance, last_trade = trader.make_one_trade(prediction,usd_balance,btc_balance,current_btc_price, comission_rate, last_trade, leverage)        
        # Calculates how much usd would he have if he closed trade
        after_close_usd_balance, _ = trader.close_trade(usd_balance, btc_balance, current_btc_price, comission_rate)
        # Updates stats
        calculate_stats(last_trade, after_close_usd_balance, last_balance)
        last_balance = after_close_usd_balance       
        
        # Waits minute
        time.sleep(60)
def calculate_stats(last_trade, after_close_usd_balance, last_balance):
    global long_count, short_count, hold_count, good_trade_count, bad_trade_count , good_trade_profit, bad_trade_loss
    if last_trade == 'long':
        long_count += 1
    elif last_trade == 'close':
        last_trade += 1
    else:
        hold_count +=1
    if after_close_usd_balance > last_balance:
        good_trade_count +=1
        good_trade_profit += (after_close_usd_balance - last_balance)        
    else:
        bad_trade_count +=1
        bad_trade_loss += (last_balance - after_close_usd_balance)

if __name__ == '__main__':
    initialize_start_balance()
    start_trading()
    train_loop()