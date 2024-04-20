# This script is for simulating trading on live data (forward testing)
import time
import threading
import best_brain as bb
import binance_data_fetcher as bdf
from data_manager import LoaderOHLCV
import trader
import json_data_handler
import csv_data_handler

symbol = 'BTCUSDT'
look_back = bb.look_back
# loads trading data
bb.load_model()
bb.model.to(bb.device)

class TradingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        global td, start_time
        td = json_data_handler.load_trading_data()
        # saves to json : is_trading = True
        json_data_handler.update_trading_data("is_trading", True)
        # sets the initial value, so it can make the first trade properly
        last_trade = 'hold'
        # actions that are needed to calculate upcoming statistics
        raw_data = bdf.get_live_minute_datapoints(symbol, lookback = 1)
        current_btc_price = float(raw_data['Close'].iloc[-1])
        try:
        last_balance,_ = trader.close_trade(td["USD_balance"], td["BTC_balance"], current_btc_price, td["commission_rate"])
        DataManager =  LoaderOHLCV(look_back, mode= bb.load_data_mode)
        while not self._stop_event.is_set():
            # Meausures starting time 
            start_time = time.time()
            last_balance, last_trade = self.one_trading_loop_iteration (last_balance, last_trade, DataManager)
            # Mechanism how to stop as fast as possible (can not have jsut time.sleep(60))
            intervals = 0
            while intervals < 60 and not self._stop_event.is_set():
                intervals += 1 
                time.sleep(1)
                if intervals % 5 == 0 :
                    # Calculates time from begging of the oone iteration of trading loop
                    time_spend = round(time.time() - start_time)
                    # Saves the time spend
                    self.save_time_spent(time_spend)
        # Closes all trades + sets is trading to False 
        self.after_stop(last_balance)
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
        td["USD_balance"], td["BTC_balance"], last_trade = trader.make_one_trade(prediction,td["USD_balance"],td["BTC_balance"],current_btc_price,td["commission_rate"], last_trade, td["leverage"])        
        # Calculates how much usd would he have if he closed trade
        after_close_usd_balance, _ = trader.close_trade(td["USD_balance"], td["BTC_balance"], current_btc_price, td["commission_rate"])
        # Updates stats
        self.calculate_stats(last_trade, after_close_usd_balance, last_balance)
        # Saves all stats
        self.save_all_trading_data()
        # Saves balance history to csv
        self.save_balance_history(after_close_usd_balance, raw_data["Date"].iloc[-1]) # raw_data["Date"].iloc[-1] = date of the close price we make trade for
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
        if after_close_usd_balance >= last_balance:
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
    # function to save just time, so it does not save not updated values
    def save_time_spent(self, time_spent_on_one_loop):
        new_time = time_spent_on_one_loop + td["time_spent_trading"]
        json_data_handler.update_trading_data(key="time_spent_trading",value=new_time)
    # Saves usd balance that it would have after closing all trades + saves timestamp in seconds 
    def save_balance_history(self, usd_balance, timestamp_date):
        # Converts date timestamp to timestamp of seconds
        timestamp = timestamp_date.timestamp()
        csv_data_handler.append_row_to_csv(usd_balance, timestamp)
    # Function that is called after stopping a trading loop
    # It closes last trade
    def after_stop(self, las_balance_after_close):
        # This is basicaly closing trade but without calling the function (since there are not present the required params)
        # remembers the last_balance (what balance will it have after close)
        usd_balance_after_stop = las_balance_after_close
        # Sets the btc balance to 0
        btc_balance_after_stop = 0
        json_data_handler.update_trading_data("USD_balance", usd_balance_after_stop)
        json_data_handler.update_trading_data("BTC_balance", btc_balance_after_stop)
        # Saves to json: is_trading = False
        json_data_handler.update_trading_data("is_trading", False)
def start_trading():
    print("starting trading loop")
    global trading_thread
    trading_thread = TradingThread()
    trading_thread.start()
def stop_trading():
    print("stopping trading loop")
    trading_thread.stop()
    trading_thread.join()