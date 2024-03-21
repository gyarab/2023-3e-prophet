from flask import Flask, render_template, jsonify
from binance_data_fetcher import get_current_btc_value

app = Flask(__name__)

# Initial BTC value
btc_value = 0

@app.route('/')
def index():
    
    global btc_value
    btc_value = get_current_btc_value()
    return render_template('prophet.html', btc_value=btc_value)

@app.route('/increment_btc_value')
def increment_btc_value():
    return render_template('prophet.html')
    

#if this script is being run directly, run it in debbug mode( detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)