import best_brain as bb
import binance_data_fetcher
from data_manager import LoaderOHLCV
import torch
import matplotlib.pyplot as plt

device = bb.device
model = bb.model
model_path = bb.model_path
# Initialize weights values
model = bb.load_data_model(model,model_path)
model.to(device)
# prepares data
DataManager = LoaderOHLCV(bb.look_back, bb.features_columns, bb.load_data_mode,'Back_test_1_minute.csv')
raw_data = DataManager.load_data_from_csv()
prepared_data = DataManager.prepare_dataframe_for_lstm3(raw_data, train= False)
prepared_data_as_np = prepared_data.to_numpy()
def create_back_test_graph(algo_usd_balance_history, bh_usd_balance_history): #bh = buy and hold
    print('Creating graph')
    plt.plot(algo_usd_balance_history,color = 'red')
    plt.plot(bh_usd_balance_history, color = 'blue')
    plt.show()
def calculate_win_rate(trades_taken_wrongly, amount_of_trades):
    loose_ratio = trades_taken_wrongly / amount_of_trades
    loose_rate = loose_ratio * 100
    win_rate = 100 - loose_rate
    return win_rate
def balance_after_commission(usd_balance, btc_balance, commission_rate = 0.05, Buy = True):
    if Buy:
        btc_commission = btc_balance * (commission_rate / 100) 
        btc_balance = btc_balance - btc_commission
    else: # Sell scenario 
        usd_commission = usd_balance * (commission_rate / 100)
        usd_balance = usd_balance - usd_commission
    return usd_balance, btc_balance
def sellAllBtc(usd_balance, btc_balance, current_btc_price, use_commissions):
    usd_will_get = btc_balance * current_btc_price
    usd_balance += usd_will_get
    btc_balance = 0
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
# Opens long position
def long_position(usd_balance, btc_balance,amount_of_balanace_per_trade,current_btc_price, use_commissions):
    valueu_of_btc_balance = btc_balance * current_btc_price
    usd_to_buy_with = valueu_of_btc_balance + usd_balance * amount_of_balanace_per_trade
    btc_bought = usd_to_buy_with / current_btc_price
    
    usd_balance -= usd_to_buy_with 
    btc_balance += btc_bought
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=True)
    return usd_balance, btc_balance
# Opens short position
def short_position(usd_balance, btc_balance,amount_of_balanace_per_trade,current_btc_price, use_commissions):    
    btc_to_buy_with = btc_balance * amount_of_balanace_per_trade
    usd_bought = btc_to_buy_with * current_btc_price
    
    usd_balance += usd_bought 
    btc_balance -= btc_to_buy_with
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
def make_one_trade(prediction, usd_balance, btc_balance, current_btc_price, use_commissions, last_trade):
    # how much of portfolio will be traded in one trade
    amount_of_balanace_per_trade = 1
    # converts the prediction to percentage - how likely will btc go up
    prediction_in_percentage = prediction * 100
    #usd_balance, btc_balance = sellAllBtc(usd_balance,btc_balance,current_btc_price, use_commissions)
    # Opens long position
    if prediction_in_percentage > 50 and last_trade == 'short' :
        usd_balance, btc_balance = long_position(usd_balance, btc_balance,amount_of_balanace_per_trade,current_btc_price, use_commissions)
        last_trade = 'long'
    # Opens short position - sells what I dont havem gets negative btc balance
    elif prediction_in_percentage < 50 and last_trade == 'long':
        usd_balance, btc_balance = short_position(usd_balance, btc_balance,amount_of_balanace_per_trade,current_btc_price, use_commissions)
        last_trade = 'short'
    return usd_balance, btc_balance, last_trade
# Buys btc wih all btc - suitable for buy and hold simulation
def initialize_bh_balance(current_btc_price, usd_balance = 10000):
    btc_bought = usd_balance /current_btc_price # calculates amount of btc to buy
    btc_balance = btc_bought
    usd_balance = 0 
    return usd_balance, btc_balance
def get_btc_price_for_current_sequence(index_of_data_sequence):
    index_of_raw_data = index_of_data_sequence + bb.look_back
    current_btc_price = raw_data.loc[index_of_raw_data, 'Close']
    return float(current_btc_price)
def back_test_loop(use_commissions = True):
    print('Starting back testing loop')
    usd_balance = 10000
    btc_balance = 0
    last_usd_balance = 0
    trades_taken_wrongly = 0
    print (f'Starting with usd balance: {usd_balance}')
    # Create tqdm instance
    current_btc_price = 0
    bh_usd_balance, bh_btc_balance = initialize_bh_balance(get_btc_price_for_current_sequence(0)) # gets starting price of btc
    usd_balance_history = []
    bh_usd_balance_history = []
    long_count = 0
    short_count = 0
    last_trade = None
    for index, one_sequence in enumerate(prepared_data_as_np):
        current_btc_price = get_btc_price_for_current_sequence(index)
        usd_balance_after_sell,_ = sellAllBtc(usd_balance,btc_balance,current_btc_price, use_commissions)
        bh_usd_balance_test, _ = sellAllBtc(bh_usd_balance, bh_btc_balance, current_btc_price, use_commissions)
        
        usd_balance_history.append(usd_balance_after_sell) # Remembers usd balance of algo
        bh_usd_balance_history.append(bh_usd_balance_test) # Remembers usd balance of buy and hold
        if (last_trade == 'long'):
            long_count += 1
        elif (last_trade == 'short'):
            short_count += 1
        
        if( last_usd_balance > usd_balance_after_sell):
            trades_taken_wrongly += 1
        last_usd_balance = usd_balance_after_sell
        
        # Tranforms sequence to correct form
        one_sequence = one_sequence.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        
        prediction = bb.make_one_prediction(one_sequence_tensor)
        
        usd_balance, btc_balance, last_trade = make_one_trade(prediction,usd_balance,btc_balance,current_btc_price, use_commissions, last_trade)
        
    usd_balance, btc_balance = sellAllBtc(usd_balance,btc_balance,current_btc_price, use_commissions)
    print(f'usd balance ended with: {usd_balance}')
    print(f'win rate: {calculate_win_rate(trades_taken_wrongly, long_count + short_count)} %')
    print(f'made {long_count} longs')
    print(f'made {short_count} shorts')
    create_back_test_graph(usd_balance_history, bh_usd_balance_history)        
if __name__ == '__main__':
    back_test_loop(False)