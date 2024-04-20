# This cript is used for reading and writting into the csv file that stores usd_balance history

import csv
import os
from datetime import datetime
import json

filename = "balance_history.csv"
json_filename = "trade_data.json"
data_folder = "data"
file_path = os.path.join(data_folder, filename)
# Check if the data folder exists and create it if it doesn't
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def append_row_to_csv(usd_balance, timestamp):
    # Open the CSV file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the new row to the CSV file
        csv_writer.writerow([usd_balance, timestamp])
        
def read_history():
    data_folder = "data"
    filepath = os.path.join(data_folder, filename)
    
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File '{filename}' not found.")
        return []
    
    with open(filepath, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        lines = list(csv_reader)
    return lines


def prepare_array():
    history = read_history()
    current_timestamp = datetime.now()
    
    

    print("start :------")
    # Initialize an array of 60 values representing each minute
    balance_array = [0] * 60

    if not any(history):
        print("nig")
        # If history is empty, return balance_array filled with the last known non-zero balance
        last_non_zero_balance = 10000  # You might want to adjust this default value
        balance_array = [last_non_zero_balance] * 60
        return balance_array


    for line in history:
        timestamp = datetime.fromtimestamp(float(line[1]))
        balance = float(line[0])
        # Calculate the difference in minutes between the current timestamp and the timestamp from history
        minutes_diff = (current_timestamp - timestamp).total_seconds() // 60
        # If the difference is within the last hour
        if 0 <= minutes_diff < 60:
            # Set the balance at the corresponding minute index
            minute_index = int(59 - minutes_diff)  # reverse index since we start from 60th minute
            balance_array[minute_index] = balance
    

    #No trading in last 60 mins and csv not empty
    if all(val == 0 for val in balance_array):
        balance_array = [float(history[-1][0])] * 60
        return balance_array
    
    # Find the index of the first non-zero value
    first_non_zero_index = next((i for i, val in enumerate(balance_array) if val != 0), None)

    # Replace zeros from start until the first non-zero value with that value
    if first_non_zero_index is not None:
        for i in range(first_non_zero_index):
            balance_array[i] = balance_array[first_non_zero_index]   

    # Replace every remaining zero with the last non-zero value
    last_non_zero = None
    for i in range(len(balance_array)):
        if balance_array[i] != 0:
            last_non_zero = balance_array[i]
        else:
            balance_array[i] = last_non_zero    

    return balance_array



def dummy_numbers():
    now = datetime.now()
    timestamp_float = now.timestamp()
    formatted_timestamp = "{:.0f}".format(timestamp_float)
    return "99999999999," + formatted_timestamp

    

# Creates new blank csv file (not reseting)
def reset_history():
    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csvfile:
        pass  # Simply open and close the file to create an empty CSV file

if __name__ == '__main__':
    print(dummy_numbers()) 
    #print(datetime.now())
    print(prepare_array()) 
    #reset_history()