# Thi scripts is for testing the trading algorythm on historical data
import best_brain as bb
import binance_data_fetcher
from data_manager import LoaderOHLCV
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc

device = bb.device
model = bb.model
model_path = bb.model_path
# Initialize weights values
model = bb.load_data_model(model,model_path)
model.to(device)
# prepares data
DataManager = LoaderOHLCV(bb.look_back, bb.features_columns, bb.load_data_mode,'Backtest_1_minute.csv')
raw_data = DataManager.load_data_from_csv()
prepared_data = DataManager.prepare_dataframe_for_lstm3(raw_data, train= False)
prepared_data_as_np = prepared_data.to_numpy()
prepared_data_as_np = dc(np.flip(prepared_data_as_np, axis= 1)) # converts the data to correct chronological order
def create_back_test_graph(algo_usd_balance_history, bh_usd_balance_history, sh_usd_balance_history): #bh = buy and hold
    print('Creating graph')
    plt.plot(bh_usd_balance_history, color = 'blue')
    plt.plot(sh_usd_balance_history, color = 'green')
    plt.plot(algo_usd_balance_history,color = 'red')
    plt.show()
def calculate_win_rate(bad_trade_count, amount_of_trades):
    loose_ratio = bad_trade_count / amount_of_trades
    loose_rate = loose_ratio * 100
    win_rate = 100 - loose_rate
    return win_rate
def calculate_gain_to_loss(usd_gained, usd_lost, good_trade_count, bad_trade_count):
    avrage_gain = usd_gained / good_trade_count
    avrage_loss = usd_lost / bad_trade_count
    gain_to_loss = avrage_gain / avrage_loss # how many dollars were gained for each lost, on avrage
    return gain_to_loss
def balance_after_commission(usd_balance, btc_balance, commission_rate = 0.05, Buy = True):
    if Buy:
        btc_commission = btc_balance * (commission_rate / 100) 
        btc_balance = btc_balance - btc_commission
    else: # Sell scenario 
        usd_commission = usd_balance * (commission_rate / 100)
        usd_balance = usd_balance - usd_commission
    return usd_balance, btc_balance
def close_trade(usd_balance, btc_balance, current_btc_price, use_commissions):
    usd_will_get = btc_balance * current_btc_price
    usd_balance += usd_will_get
    btc_balance = 0
    if use_commissions:
        usd_balance, btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
# Opens long position
def long_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions):
    usd_to_buy_with = usd_balance * leverage
    btc_bought = usd_to_buy_with / current_btc_price
    
    usd_balance -= usd_to_buy_with 
    btc_balance += btc_bought
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=True)
    return usd_balance, btc_balance
# Opens short position
def short_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions):    
    usd_to_sell_with = usd_balance * leverage
    btc_sold = usd_to_sell_with / current_btc_price

    usd_balance += usd_to_sell_with 
    btc_balance -= btc_sold
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
# decides what trade to do, if any
def make_one_trade(prediction, usd_balance, btc_balance, current_btc_price, use_commissions, last_trade, leverage):
    # Opens long position
    if prediction < 0.5 and btc_balance <= 0: # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance, btc_balance, current_btc_price, use_commissions)
        usd_balance, btc_balance = long_position(usd_balance, btc_balance, leverage, current_btc_price, use_commissions)
        last_trade = 'long'
    #Opens short position - sells what I dont havem gets negative btc balance
    elif prediction > 0.5 and btc_balance >= 0: # # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance,btc_balance,current_btc_price, use_commissions)
        usd_balance, btc_balance = short_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions)
        last_trade = 'short'
    else:
        last_trade = 'hold'
    return usd_balance, btc_balance, last_trade
# Buys btc wih all btc - suitable for buy and hold simulation
def initialize_bh_balance(current_btc_price, usd_balance = 10000):
    btc_bought = usd_balance /current_btc_price # calculates amount of btc to buy
    btc_balance = btc_bought
    usd_balance = 0 
    return usd_balance, btc_balance
def initialize_sh_balance (current_btc_price, usd_balance = 10000):
    btc_sold = usd_balance /current_btc_price # calculates amount of btc to buy
    btc_balance = 0 - btc_sold
    usd_balance += btc_sold * current_btc_price 
    return usd_balance, btc_balance
def get_btc_price_for_current_sequence(index_of_data_sequence):
    index_of_raw_data = index_of_data_sequence + bb.look_back
    current_btc_price = raw_data.loc[index_of_raw_data, 'Close']
    return float(current_btc_price)
def back_test_loop(start_usd_balance = 10000, leverage = 1, use_commissions = True):
    print('Starting back testing loop')
    usd_balance = start_usd_balance
    btc_balance = 0
    last_usd_balance = 0
    print (f'Starting with usd balance: {usd_balance}')
    # Create tqdm instance
    current_btc_price = 0
    bh_usd_balance, bh_btc_balance = initialize_bh_balance(get_btc_price_for_current_sequence(0), usd_balance= usd_balance) # gets starting price of btc
    sh_usd_balance, sh_btc_balance = initialize_sh_balance(get_btc_price_for_current_sequence(0), usd_balance= usd_balance)
    usd_balance_history = []
    bh_usd_balance_history = []
    sh_usd_balance_history = []
    long_count = 0
    short_count = 0
    hold_count = 0
    bad_trade_count = 0
    good_trade_count = 0
    usd_gained = 0
    usd_lost = 0
    last_trade = None
    for index, one_sequence in enumerate(prepared_data_as_np):
        current_btc_price = get_btc_price_for_current_sequence(index)
        usd_balance_after_close,_ = close_trade(usd_balance,btc_balance,current_btc_price, use_commissions)
        bh_usd_balance_test, _ = close_trade(bh_usd_balance, bh_btc_balance, current_btc_price, use_commissions)
        sh_usd_balance_test, _ = close_trade(sh_usd_balance,sh_btc_balance,current_btc_price, use_commissions)
        usd_balance_history.append(usd_balance_after_close) # Remembers usd balance of algo
        bh_usd_balance_history.append(bh_usd_balance_test) # Remembers usd balance of buy and hold
        sh_usd_balance_history.append(sh_usd_balance_test) # Remembers usd balance of short and hold
        if (last_trade == 'long'):
            long_count += 1
        elif (last_trade == 'short'):
            short_count += 1
        else:
            hold_count +=1
        
        if( last_usd_balance > usd_balance_after_close):
            bad_trade_count += 1
            usd_lost = last_usd_balance - usd_balance_after_close
        else:
            good_trade_count += 1
            usd_gained = usd_balance_after_close - last_usd_balance
        last_usd_balance = usd_balance_after_close
        
        # Tranforms sequence to correct form
        one_sequence = one_sequence.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        
        prediction = bb.make_one_prediction(one_sequence_tensor)
        
        usd_balance, btc_balance, last_trade = make_one_trade(prediction,usd_balance,btc_balance,current_btc_price, use_commissions, last_trade, leverage)
        
    usd_balance, btc_balance = close_trade(usd_balance,btc_balance,current_btc_price, use_commissions)
    usd_balance_history.append(usd_balance)
    print(f'usd balance ended with: {usd_balance}')
    print(f'per dollar gained: {usd_balance / start_usd_balance}') # should be divided by start balance
    print(f'win rate: {calculate_win_rate(bad_trade_count, hold_count + long_count + short_count)} %')
    print(f'gain to loss on avrage: {calculate_gain_to_loss(usd_gained, usd_lost, good_trade_count, bad_trade_count)}')
    print(f'good trades: {good_trade_count}')
    print(f'bad trades: {bad_trade_count}')
    print(f'longs: {long_count}')
    print(f'shorts: {short_count}')
    print(f'holds: {hold_count}')
    create_back_test_graph(usd_balance_history, bh_usd_balance_history, sh_usd_balance_history)        
if __name__ == '__main__':
    back_test_loop(use_commissions = False)