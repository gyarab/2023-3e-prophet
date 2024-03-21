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
DataManager = LoaderOHLCV(bb.look_back, bb.features_columns, bb.load_data_mode,'Back_test.csv')
raw_data = DataManager.load_data_from_csv()
prepared_data = DataManager.prepare_dataframe_for_lstm3(raw_data, train= False)
prepared_data_as_np = prepared_data.to_numpy()
def create_back_test_graph(algo_usd_balance_history, bh_usd_balance_history): #bh = buy and hold
    plt.plot(algo_usd_balance_history,color = 'red')
    plt.plot(bh_usd_balance_history, color = 'blue')
    plt.show()
def make_one_prediction(one_sequence_tensor):
    one_sequence_tensor = one_sequence_tensor.to(device)
    # Model makes prediction
    with torch.no_grad():
                model.eval()
                prediction = model(one_sequence_tensor)
                prediction_values = prediction.item()
    return prediction_values
def calculate_win_rate(trades_taken_wrongly, amount_of_trades):
    loose_ratio = trades_taken_wrongly / amount_of_trades
    loose_rate = loose_ratio * 100
    win_rate = 100 - loose_rate
    return win_rate
def balance_after_comission(usd_balance, btc_balance, comission_rate = 0.1, Buy = True):
    comission = usd_balance * (comission_rate / 100) 
    if Buy:
        btc_comission = btc_balance * (comission_rate / 100) 
        btc_balance = btc_balance - btc_comission
    else: # Sell scenario 
        usd_comission = usd_balance * (comission_rate / 100)
        usd_balance = usd_balance - usd_comission
    return usd_balance, btc_balance
def sellAllBtc(usd_balance, btc_balance, current_btc_price):
    usd_will_get = btc_balance * current_btc_price
    usd_balance += usd_will_get
    btc_balance = 0
    usd_balance,btc_balance = balance_after_comission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
# Opens long position
def long_position(usd_balance, btc_balance,percentage_per_transaction,current_btc_price):
    usd_to_buy_with = usd_balance * percentage_per_transaction
    btc_bought = usd_to_buy_with / current_btc_price
    
    usd_balance -= usd_to_buy_with 
    btc_balance += btc_bought
    usd_balance,btc_balance = balance_after_comission(usd_balance,btc_balance, Buy=True)
    return usd_balance, btc_balance
# Opens short position
def short_position(usd_balance, btc_balance,percentage_per_transaction,current_btc_price):
    usd_to_sell_with = usd_balance * percentage_per_transaction
    btc_sold = usd_to_sell_with / current_btc_price
    
    usd_balance += usd_to_sell_with 
    btc_balance -= btc_sold
    usd_balance,btc_balance = balance_after_comission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
def make_one_trade(prediction, usd_balance, btc_balance, current_btc_price):
    # sells all btc, so the algo is now trading just with the set amout of portfolio
    usd_balance, btc_balance = sellAllBtc(usd_balance,btc_balance,current_btc_price)
    # how much of portfolio will be traded in one trade
    percentage_per_transaction = 1
    # converts the prediction to percentage - how likely will btc go up
    prediction_in_percentage = prediction * 100
    # Opens long position
    if prediction_in_percentage > 51.5:
        usd_balance, btc_balance = long_position(usd_balance, btc_balance,percentage_per_transaction,current_btc_price)
    # Opens short position - sells what I dont havem gets negative btc balance
    elif prediction_in_percentage < 48.5:
        usd_balance, btc_balance = short_position(usd_balance, btc_balance,percentage_per_transaction,current_btc_price)
        
    return usd_balance, btc_balance
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
def back_test_loop():
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
    for index, one_sequence in enumerate(prepared_data_as_np):
        current_btc_price = get_btc_price_for_current_sequence(index)
        usd_balance_after_sell,_ = sellAllBtc(usd_balance,btc_balance,current_btc_price)
        bh_usd_balance_test, _ = sellAllBtc(bh_usd_balance, bh_btc_balance, current_btc_price)
        
        usd_balance_history.append(usd_balance_after_sell) # Remembers usd balance of algo
        bh_usd_balance_history.append(bh_usd_balance_test) # Remembers usd balance of buy and hold
        if( last_usd_balance > usd_balance_after_sell):
            trades_taken_wrongly += 1
        last_usd_balance = usd_balance_after_sell
        
        # Tranforms sequence to correct form
        one_sequence = one_sequence.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        
        prediction = make_one_prediction(one_sequence_tensor)
        
        usd_balance, btc_balance = make_one_trade(prediction,usd_balance,btc_balance,current_btc_price)
        
    usd_balance, btc_balance = sellAllBtc(usd_balance,btc_balance,current_btc_price)
    print(f'usd balance ended with: {usd_balance}')
    print(f'win rate: {calculate_win_rate(trades_taken_wrongly, len(prepared_data_as_np))} %')
    create_back_test_graph(usd_balance_history, bh_usd_balance_history)        
if __name__ == '__main__':
    back_test_loop()