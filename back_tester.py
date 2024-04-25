# Thi scripts is for testing the trading algorythm on historical data
import best_brain as bb
from data_manager import LoaderOHLCV
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc
import trader

device = bb.device
# Initialize weights values
model_name = "historical_model.pth"
bb.model_path = bb.create_model_path(model_name)
bb.load_model()
bb.model.to(device)
# prepares data
DataManager = LoaderOHLCV(bb.look_back, bb.load_data_mode,'Backtest_1_minute.csv')
raw_data = DataManager.load_data_from_csv()

prepared_data = DataManager.prepare_dataframe_for_lstm2(raw_data, train= False)
prepared_data_as_np = prepared_data.to_numpy()
# converts the data to correct chronological order
prepared_data_as_np = dc(np.flip(prepared_data_as_np, axis= 1))
def create_back_test_graph(algo_usd_balance_history, bh_usd_balance_history, sh_usd_balance_history): #bh = buy and hold
    print('Creating graph')
    # Plots portfolios
    plt.plot(bh_usd_balance_history, color = 'blue', label = 'Buy and Hold Balance')
    plt.plot(sh_usd_balance_history, color = 'green', label = 'Short and Hold Balance')
    plt.plot(algo_usd_balance_history,color = 'red', label = 'Bot Balance' )
    # Adding labels and title
    plt.xlabel('Minute')
    plt.ylabel('USD Balance')
    plt.title('Balance Over Time')
    plt.legend()  # Show legend with labels
    plt.grid(True)  # Add grid
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
    index_of_raw_data = index_of_data_sequence + bb.look_back + 1
    current_btc_price = raw_data.loc[index_of_raw_data, 'Close']
    return float(current_btc_price)
def back_test_loop(start_usd_balance = 10000, leverage = 1, commission_rate = 0):
    print('Starting back testing loop')
    usd_balance = start_usd_balance
    btc_balance = 0
    last_usd_balance = 0
    print (f'Starting with usd balance: {usd_balance}')
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
        usd_balance_after_close,_ = trader.close_trade(usd_balance,btc_balance,current_btc_price, commission_rate)
        bh_usd_balance_test, _ = trader.close_trade(bh_usd_balance, bh_btc_balance, current_btc_price, commission_rate)
        sh_usd_balance_test, _ = trader.close_trade(sh_usd_balance,sh_btc_balance,current_btc_price, commission_rate)
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
        
        usd_balance, btc_balance, last_trade = trader.make_one_trade(prediction,usd_balance,btc_balance,current_btc_price, commission_rate, last_trade, leverage)
        
    usd_balance, btc_balance = trader.close_trade(usd_balance,btc_balance,current_btc_price, commission_rate)
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
    back_test_loop(commission_rate = 0.00, leverage= 1)