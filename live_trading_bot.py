# This script is for simulating trading on live data (forward testing)
import time
from copy import deepcopy as dc
import numpy as np
import torch
import best_brain as bb
import binance_data_fetcher as bdf
from data_manager import LoaderOHLCV
is_trading = False

symbol = 'BTCUSDT'
usd_balance = 0
btc_balance = 0
leverage = 0
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
        usd_balance, btc_balance, last_trade = make_one_trade(prediction,current_btc_price, comission_rate, last_trade, leverage)        
        # waits minute
        time.sleep(60)
def make_one_trade(prediction, current_btc_price, comission_rate, last_trade, leverage):
    global usd_balance
    global btc_balance
    # Opens long position
    if prediction < 0.5 and btc_balance <= 0: # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance, btc_balance, current_btc_price, comission_rate)
        usd_balance, btc_balance = long_position(usd_balance, btc_balance, leverage, current_btc_price, comission_rate)
        last_trade = 'long'
    #Opens short position - sells what I dont havem gets negative btc balance
    elif prediction > 0.5 and btc_balance >= 0: # # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance,btc_balance,current_btc_price, comission_rate)
        usd_balance, btc_balance = short_position(usd_balance, btc_balance,leverage,current_btc_price, comission_rate)
        last_trade = 'short'
    else:
        last_trade = 'hold'
    return usd_balance, btc_balance, last_trade
def long_position(usd_balance, btc_balance,leverage,current_btc_price, comission_rate):
    usd_to_buy_with = usd_balance * leverage
    btc_bought = usd_to_buy_with / current_btc_price
    
    usd_balance -= usd_to_buy_with 
    btc_balance += btc_bought
    if comission_rate > 0:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance,comission_rate, Buy=True)
    return usd_balance, btc_balance
def short_position(usd_balance, btc_balance,leverage,current_btc_price, comission_rate):    
    usd_to_sell_with = usd_balance * leverage
    btc_sold = usd_to_sell_with / current_btc_price

    usd_balance += usd_to_sell_with 
    btc_balance -= btc_sold
    if comission_rate > 0:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance,comission_rate, Buy=False)
    return usd_balance, btc_balance
def close_trade(usd_balance, btc_balance, current_btc_price, comission_rate):
    usd_will_get = btc_balance * current_btc_price
    usd_balance += usd_will_get
    btc_balance = 0
    if comission_rate > 0:
        usd_balance, btc_balance = balance_after_commission(usd_balance,btc_balance,comission_rate, Buy=False)
    return usd_balance, btc_balance
def balance_after_commission(usd_balance, btc_balance, commission_rate = 0, Buy = True):
    if Buy:
        btc_commission = btc_balance * (commission_rate / 100) 
        btc_balance = btc_balance - btc_commission
    else: # Sell scenario 
        usd_commission = usd_balance * (commission_rate / 100)
        usd_balance = usd_balance - usd_commission
    return usd_balance, btc_balance
usd_balance, btc_balance = initialize_start_balance()
start_trading()
train_loop()