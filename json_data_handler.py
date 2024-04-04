import json
from binance_data_fetcher import get_current_btc_value
import os
trade_data_file_name = 'data/trade_data.json'
def update_trading_data(filename=trade_data_file_name, **kwargs):
    # Ensure that the data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Fetch current BTC value
    btc_value = get_current_btc_value()
    
    # Load existing data if the file exists
    try:
        with open(filename, 'r') as file:
            trading_data = json.load(file)
    except FileNotFoundError:
        trading_data = {}

    # Update trading data with new values
    trading_data.update(kwargs)
    trading_data["Btc_valueat_close_app"] = btc_value

    # Open the file in write mode and save the updated data as JSON
    with open(filename, 'w') as file:
        json.dump(trading_data, file, indent=4)


def load_trading_data(filename=trade_data_file_name, data_key=None):
    try:
        with open(filename, 'r') as file:
            trading_data = json.load(file)
        
        # If data_key is specified, return only the value corresponding to that key
        if data_key is not None:
            if data_key in trading_data:
                return {data_key: trading_data[data_key]}
            else:
                print(f"Key '{data_key}' not found in the trading data.")
                return None
        
        # If data_key is not specified, return the entire trading data
        return trading_data
    
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{filename}'.")
        return None
    

def reset_saved_data(filename=trade_data_file_name):
    # Fetch current BTC value
    btc_value = get_current_btc_value()
    
    # Define trading data with fetched BTC value
    trading_data = {
        "Time_spent_trading": 0,
        "Btc_value_at_close_app": btc_value,
        "USD_balance": 10000,
        "BTC_balance": 0,
        "long_count": 0,
        "short_count": 0,
        "hold_count": 0,
        "bad_trade_count": 0,
        "good_trade_count": 0,
        "total_profit": 0,
        "total_loss": 0,
        "leverage" : 1,
        "comission_rate" : 0
    }
    # Ensure that the data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(trading_data, file, indent=4)