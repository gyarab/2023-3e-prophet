import ccxt

def get_btc_price():
    
    platform = ccxt.bingx()
    # Fetch ticker information for BTC/USDT (you can change the symbol based on your needs)
    ticker = platform.fetch_ticker('BTC/USDT')
    # Extract the last price from the ticker data
    last_price = ticker['last']
    
    return last_price

if __name__ == '__main__':
    print(get_btc_price())
