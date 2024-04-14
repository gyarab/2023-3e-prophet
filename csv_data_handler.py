# This cript is used for reading and writting into the csv file that stores usd_balance history

import csv
import os
filename = "balance_history.csv"
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
        
def read_history(filename):
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
# Creates new blank csv file (not reseting)
def reset_history():
    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csvfile:
        pass  # Simply open and close the file to create an empty CSV file
