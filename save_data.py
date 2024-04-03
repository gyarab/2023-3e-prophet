import json

def save_trading_data(time_spent_trading, btc_value_invested, filename):
    # Create a dictionary to store the trading data
    trading_data = {
        "Time Spent Trading": time_spent_trading,
        "BTC Value Invested": btc_value_invested
    }

    # Open the file in write mode and save the data as JSON
    with open(filename, 'w') as file:
        json.dump(trading_data, file, indent=4)


def load_trading_data(filename):
    try:
        # Open the file in read mode and load the JSON data
        with open(filename, 'r') as file:
            trading_data = json.load(file)
        return trading_data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{filename}'.")
        return None


