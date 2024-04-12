# This script is for simulating trading on live data (forward testing)
import time
import threading
import best_brain as bb
import binance_data_fetcher as bdf
from data_manager import LoaderOHLCV
import trader
import json_data_handler

symbol = 'BTCUSDT'
look_back = 9
# loads trading data
bb.model = bb.load_data_model(bb.model, bb.model_path)
bb.model.to(bb.device)

class TradingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        global td, start_time
        td = json_data_handler.load_trading_data()
        # sets the initial value, so it can make the first trade properly
        last_trade = 'hold'
        start_time = time.time()
        # actions that are needed to calculate upcoming statistics
        raw_data = bdf.get_live_minute_datapoints(symbol, lookback = 1)
        current_btc_price = float(raw_data['Close'].iloc[-1])
        last_balance,_ = trader.close_trade(td["USD_balance"], td["BTC_balance"], current_btc_price, td["comission_rate"])
        DataManager =  LoaderOHLCV(look_back,['Close'], mode= 3) # !!! HARDCODED
        while not self._stop_event.is_set():
            last_balance, last_trade = self.one_trading_loop_iteration (last_balance, last_trade, DataManager)
            # Mechanism how to stop as fast as possible
            intervals = 0
            while intervals < 60 and not self._stop_event.is_set():
                intervals += 1 
                time.sleep(1)

    def stop(self):
        self._stop_event.set()
    def one_trading_loop_iteration(self,last_balance, last_trade, DataManager):
        global td
        td = json_data_handler.load_trading_data()
        # Gets raw data from data_fetcher
        raw_data = bdf.get_live_minute_datapoints(symbol, lookback = look_back)
        # Get the last value - the most actual BTC price 
        current_btc_price = float(raw_data['Close'].iloc[-1])
        # Prepares data
        one_sequence_tensor = DataManager.prepare_live_data(raw_data)
        # Models makes prediction
        prediction = bb.make_one_prediction(one_sequence_tensor)
        # Simulates one trade
        td["USD_balance"], td["BTC_balance"], last_trade = trader.make_one_trade(prediction,td["USD_balance"],td["BTC_balance"],current_btc_price,td["comission_rate"], last_trade, td["leverage"])        
        # Calculates how much usd would he have if he closed trade
        after_close_usd_balance, _ = trader.close_trade(td["USD_balance"], td["BTC_balance"], current_btc_price, td["comission_rate"])
        # Updates stats
        self.calculate_stats(last_trade, after_close_usd_balance, last_balance)
        # Calculates time spend on this loop
        time_spend = round(time.time() - start_time)
        td["time_spent_trading"]+= time_spend
        
        self.save_all_trading_data()
        last_balance = after_close_usd_balance
        
        return last_balance, last_trade
    def calculate_stats(self, last_trade, after_close_usd_balance, last_balance):
        global td
        if last_trade == 'long':
            td["long_count"] += 1
        elif last_trade == 'short':
            td["short_count"] += 1
        else:
            td["hold_count"] +=1
        if after_close_usd_balance > last_balance:
            td["good_trade_count"] +=1
            td["total_profit"] += (after_close_usd_balance - last_balance)        
        else:
            td["bad_trade_count"] +=1
            td["total_loss"] += (last_balance - after_close_usd_balance)
    def save_all_trading_data(self):
        print("Saving stats", end="\r") # returns the "cursor" to the same line, so it will be overwritten in the next print
        for key, value in td.items():
            # Call the function with unpacked kwargs
            json_data_handler.update_trading_data(key=key,value= value)
        print("Finished saving stats")
        
def start_trading():
    print("starting trading loop")
    global trading_thread
    trading_thread = TradingThread()
    trading_thread.start()
def stop_trading():
    print("stopping trading loop")
    trading_thread.stop()
    trading_thread.join()