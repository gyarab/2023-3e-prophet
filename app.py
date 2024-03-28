from flask import Flask, render_template, jsonify
from binance_data_fetcher import get_current_btc_value, get_last_hour_values


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
    

#if this script is being run directly, run it in debbug mode( detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)