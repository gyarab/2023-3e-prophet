from flask import Flask, render_template, jsonify, request
from binance_data_fetcher import get_current_btc_value, get_last_hour_values
from json_data_handler import save_trading_data, load_trading_data

app = Flask(__name__)

# Initial BTC value
btc_value = 0

@app.route('/')
def prophet():
    return render_template('base.html')

@app.route('/get_btc_value')
def get_btc_value():
    btc_value = get_current_btc_value()
    return str(btc_value)

@app.route('/get_last_hour_values')
def get_last_hour_values_endpoint():
    last_hour_values = get_last_hour_values()
    return {'last_hour_values': last_hour_values}

@app.route('/save_trading_data', methods=['POST'])
def save_trading_data_endpoint():
    data = request.get_json()
    save_trading_data("data.json", **data)
    return jsonify({'message': 'Trading data saved successfully'})

@app.route('/load_trading_data', methods=['POST'])
def load_trading_data_endpoint():
    trading_data = load_trading_data("data.json")  # No need to pass data_key anymore
    if trading_data is not None:
        return jsonify({'trading_data': trading_data})
    else:
        return jsonify({'message': 'Error loading trading data'})

# if this script is being run directly, run it in debug mode (detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)