from flask import Flask, render_template, jsonify, request
from binance_data_fetcher import get_current_btc_value, get_last_hour_values
from json_data_handler import update_trading_data, load_trading_data, reset_saved_data

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

@app.route('/load_trading_data', methods=['GET'])
def load_trading_data_endpoint():
    trading_data = load_trading_data()
    if trading_data is not None:
        return jsonify({'trading_data': trading_data}), 200
    else:
        return jsonify({'message': 'Error loading trading data'}), 200

@app.route('/reset_saved_data', methods=['POST'])
def reset_saved_data_endpoint():
    reset_saved_data("data.json")
    return jsonify({'message': 'Trading data has been reset.'})

@app.route('/update_trading_data', methods=['POST'])
def update_trading_data_endpoint():
    data = request.json  # Assuming you're sending JSON data to update trading data
    update_trading_data(**data)
    return jsonify({'message': 'Trading data has been updated.'})


# if this script is being run directly, run it in debug mode (detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)