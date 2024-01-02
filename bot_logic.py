import ccxt

exchange = ccxt.bingx({
        'apiKey': 'AZ8tdIuipJSfumjWyQb6ZWscupjrKXfV5IYmcoIy8xCrzI8ykI1LkyesbmjNOcg2BYK1UU9OKA4w9RqJsw',
        'secret': 'h0IMzxFfpalGFIMvV3KIjqGEDZFYZmqDmC3Bd2dgx58CyxtDyvHu52qRnkqQ9JAyjRGbFDKvUC71fq3bFGg',
    })

def get_btc_price():
    
    # Fetch ticker information for BTC/USDT (you can change the symbol based on your needs)
    ticker = exchange.fetch_ticker('BTC/USDT')
    # Extract the last price from the ticker data
    last_price = ticker['last']
    
    return last_price

def get_balance():
    
    balance = exchange.fetch_balance()

    return balance


if __name__ == '__main__':
    print(get_btc_price())
    print("CoinEx Account Balance:")
    for currency, details in get_balance()['total'].items():
        print(f"{currency}: {details}")