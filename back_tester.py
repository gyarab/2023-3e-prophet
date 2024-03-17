import best_brain as bb
import binance_data_fatcher
from data_manager import LoaderOHLCV
import torch
from tqdm import tqdm

device = bb.device
model = bb.model
model_path = bb.model_path
# Initialize weights values
model = bb.load_data_model(model,model_path)
# prepares data
raw_data = binance_data_fatcher.get_historical_data('BTCUSDT','2 Mar 2024')
DataManager = LoaderOHLCV(bb.look_back, bb.features_columns, bb.load_data_mode)
prepared_data = DataManager.prepare_dataframe_for_lstm3(raw_data, train= False)
def make_one_prediction(one_sequence_tensor):
    # Model makes prediction
    with torch.no_grad():
                model.eval()
                prediction = model(one_sequence_tensor)
                prediction_values = prediction.item()
    return prediction_values
def make_one_trade(prediction, usd_balance, btc_balance, current_btc_price):
    percentage_per_transaction = 0.02
    # converts the prediction to percentage - how likely will btc go up
    prediction_in_percentage = prediction * 100
    # buys btc with 2% of portfolio
    if prediction_in_percentage > 50:
        #print('Buying BTC with 2 percent of portfolio')
        usd_to_buy_with = usd_balance * percentage_per_transaction
        btc_bought = usd_to_buy_with / current_btc_price
        
        usd_balance -= usd_to_buy_with 
        btc_balance += btc_bought
    # sells all btc
    else:
        #print('Selling all BTC')
        usd_will_get = btc_balance * current_btc_price
        usd_balance += usd_will_get
        btc_balance = 0
        
    return usd_balance, btc_balance
def get_btc_price_for_current_sequence(index_of_data_sequence):
    index_of_raw_data = index_of_data_sequence + bb.look_back
    current_btc_price = raw_data.loc[index_of_raw_data, 'Close']
    return current_btc_price
def back_test_loop():
    print('Starting back testing loop')
    usd_balance = 100000
    btc_balance = 0
    print (f'Starting with usd balance: {usd_balance}')
    # Create tqdm instance
    progress_bar = tqdm(enumerate(prepared_data), total=len(prepared_data), desc='Back Testing')
    for index, one_sequence in progress_bar:
        # Tranforms sequence to correct form
        one_sequence = one_sequence.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        
        prediction = make_one_prediction(one_sequence_tensor)
        
        current_btc_price = get_btc_price_for_current_sequence(index)
        usd_balance, btc_balance = make_one_trade(prediction,usd_balance,btc_balance,current_btc_price)
        progress_bar.set_description(f"prediction of going up {prediction * 100}, USD {usd_balance}, BTC {btc_balance} ")
        
if __name__ == '__main__':
    back_test_loop()