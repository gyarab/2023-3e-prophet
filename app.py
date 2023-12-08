from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Initial BTC value
btc_value = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/increment_btc_value')
def increment_btc_value():
    global btc_value
    btc_value += 1
    return jsonify({'btc_value': btc_value})

#if this script is being run directly, run it in debbug mode( detailed errors + some other features)
if __name__ == '__main__':
    app.run(debug=True)