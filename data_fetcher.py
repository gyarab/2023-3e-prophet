import ccxt
import tqdm

APIKEY = '01566309D7D54F6B83CD7BD57090B485'
SECRETKEY = 'FF1A020B10EC704972C034475F2BBA140F814F87B401F37B'

EXCHANGE = ccxt.coinex({
        'apiKey': APIKEY,
        'secret': SECRETKEY,
    })

def get_btc_price():
    symbol = 'BTC/USDT'
    # Fetch ticker information for BTC/USDT pair
    ticker = EXCHANGE.fetch_ticker(symbol)

    # Extract and print the last price
    last_price = ticker['last']
    
    return last_price

def get_last_30_btc_price():
    symbol = 'BTC/USDT'
    
    # Fetch historical OHLCV data with 1-minute timeframe
    #OHLCV stands for: open, high, low, close, volume
    ohlcv = EXCHANGE.fetch_ohlcv(symbol, '1m') # Use '1m' for 1-minute timeframe

    # last 30 minutes as ?list?
    last_30_prices = ohlcv[-30:]
    
    return last_30_prices
#returns account balance as dictoniary
def get_balance():
    
    # Fetch your account balance
    balance = EXCHANGE.fetch_balance()

    return balance

def get_markets():
    markets = EXCHANGE.load_markets()

    return list(markets.keys())

if __name__ == '__main__':
    # print(get_btc_price())
    #print("CoinEx Account Balance:")
    #print(get_balance())
    
    for candle in get_last_30_btc_price():
        timestamp, open_, high, low, close, volume = candle
        print(f'Timestamp: {timestamp}, Close Price: {close}')
     