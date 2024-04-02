# This script is for simulating trading on live data (forward testing)
def start_trading():
    pass
def stop_trading():
    pass
def train_loop():
    pass
def make_one_trade():
    pass
def make_one_trade(prediction, usd_balance, btc_balance, current_btc_price, use_commissions, last_trade, leverage):
    # Opens long position
    if prediction < 0.5 and btc_balance <= 0: # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance, btc_balance, current_btc_price, use_commissions)
        usd_balance, btc_balance = long_position(usd_balance, btc_balance, leverage, current_btc_price, use_commissions)
        last_trade = 'long'
    #Opens short position - sells what I dont havem gets negative btc balance
    elif prediction > 0.5 and btc_balance >= 0: # # jsem to chce mean
        usd_balance, btc_balance = close_trade(usd_balance,btc_balance,current_btc_price, use_commissions)
        usd_balance, btc_balance = short_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions)
        last_trade = 'short'
    else:
        last_trade = 'hold'
    return usd_balance, btc_balance, last_trade
def long_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions):
    usd_to_buy_with = usd_balance * leverage
    btc_bought = usd_to_buy_with / current_btc_price
    
    usd_balance -= usd_to_buy_with 
    btc_balance += btc_bought
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=True)
    return usd_balance, btc_balance
def short_position(usd_balance, btc_balance,leverage,current_btc_price, use_commissions):    
    usd_to_sell_with = usd_balance * leverage
    btc_sold = usd_to_sell_with / current_btc_price

    usd_balance += usd_to_sell_with 
    btc_balance -= btc_sold
    if use_commissions:
        usd_balance,btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
def close_trade(usd_balance, btc_balance, current_btc_price, use_commissions):
    usd_will_get = btc_balance * current_btc_price
    usd_balance += usd_will_get
    btc_balance = 0
    if use_commissions:
        usd_balance, btc_balance = balance_after_commission(usd_balance,btc_balance, Buy=False)
    return usd_balance, btc_balance
def balance_after_commission(usd_balance, btc_balance, commission_rate = 0.05, Buy = True):
    if Buy:
        btc_commission = btc_balance * (commission_rate / 100) 
        btc_balance = btc_balance - btc_commission
    else: # Sell scenario 
        usd_commission = usd_balance * (commission_rate / 100)
        usd_balance = usd_balance - usd_commission
    return usd_balance, btc_balance