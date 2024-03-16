import best_brain as bb
import binance_data_fatcher
from data_manager import LoaderOHLCV
import torch


device = bb.device
model = bb.model
model_path = bb.model_path
# Initialize weights values
model = bb.load_data_model(model,model_path)
# prepares data
raw_data = binance_data_fatcher.get_historical_data('BTCUSDT','1 Mar 2024')
DataManager = LoaderOHLCV(bb.look_back, bb.features_columns, bb.load_data_mode)
prepared_data = LoaderOHLCV.prepare_dataframe_for_lstm3(raw_data, train= False)
def make_one_prediction(one_sequence_tensor):
    # Model makes prediction
    with torch.no_grad():
                model.eval()
                prediction = model(one_sequence_tensor)
                prediction_values = prediction.item()
    return prediction_values

def back_test_loop():
    
    usd_balance = 10000
    btc_balance = 0
    
    for one_sequence in prepared_data:
        # Tranforms sequence to correct form
        one_sequence = one_sequence.values.reshape((-1, bb.look_back * 1 + 1 , 1))
        one_sequence_tensor = torch.tensor(one_sequence).float()
        
        prediction = make_one_prediction(one_sequence_tensor)
        
        