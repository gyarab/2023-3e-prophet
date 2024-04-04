import json
from binance_data_fetcher import get_current_btc_value

def save_trading_data(filename="data.json", **kwargs):
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
    trading_data["Btc value at close app"] = btc_value

    # Open the file in write mode and save the updated data as JSON
    with open(filename, 'w') as file:
        json.dump(trading_data, file, indent=4)


def load_trading_data(filename="data.json"):
    try:
        with open(filename, 'r') as file:
            trading_data = json.load(file)
        return trading_data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{filename}'.")
        return None
    

def reset_saved_data(filename="data.json"):
    # Fetch current BTC value
    btc_value = get_current_btc_value()
    
    # Define trading data with fetched BTC value
    trading_data = {
        "Time Spent Trading": 0,
        "Btc value at close app": btc_value,
        "USD in wallet": 10000,
        "money in BTC": 0,
        "long trades count": 0,
        "short trades count": 0,
        "how many holds": 0,
        "bad trades count": 0,
        "good trades count": 0,
        "good trades profit": 0,
        "bad trades loss": 0,
        "leverage" : 1,
        "comission_rate" : 0
    }

    with open(filename, 'w') as file:
        json.dump(trading_data, file, indent=4)




