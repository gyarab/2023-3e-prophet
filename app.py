from flask import Flask, render_template, jsonify, request
from binance_data_fetcher import get_current_btc_value, get_last_hour_values
from save_data import save_trading_data, load_trading_data


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
    time_spent_trading = data['time_spent_trading']
    btc_value_invested = data['btc_value_invested']
    filename = "trading_data.json"
    save_trading_data(time_spent_trading, btc_value_invested, filename)
    return jsonify({'message': 'Trading data saved successfully'})

@app.route('/load_trading_data')
def load_trading_data_endpoint():
    filename = "trading_data.json"
    trading_data = load_trading_data(filename)
    return jsonify({'trading_data': trading_data})

    

#if this script is being run directly, run it in debbug mode( detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)