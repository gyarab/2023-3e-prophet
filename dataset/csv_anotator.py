# this script servers as creator of training data
# note that u have to have a "data" folder relative to this script - the script reads from it and writes into it

import csv # lib for csv files access
from tqdm import tqdm # loop progress bar in terminal
import os # file paths
import pandas as pd
import ta

def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    return input_file_path


# adds anotation off technical indicators and target values and their differences
def fully_anotate(csv_file, output_file):
    
    input_file_path = get_absolute_path(csv_file)
    output_file_path = get_absolute_path(output_file)
    # Load your OHLCV CSV file
    df = pd.read_csv(input_file_path)

    # Convert the 'timestamp' column to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the dataframe by timestamp in ascending order
    df = df.sort_values(by='timestamp')

    # Add technical indicators
    df['ema_14'] = ta.trend.EMAIndicator(close=df['close'], window=14).ema_indicator()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_upper'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_lower'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['ichimoku_a'] = ta.trend.IchimokuIndicator(high=df['high'], low=df['low']).ichimoku_a()
    df['ichimoku_b'] = ta.trend.IchimokuIndicator(high=df['high'], low=df['low']).ichimoku_b()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(close=df['close'], high=df['high'], low=df['low']).williams_r()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    
    # Add target_value and target_value_difference columns
    df['target_value'] = df['close'].shift(-1)  # Next close value
    df['target_value_difference'] = df['target_value'] - df['close']  # Difference between next close and current close
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Save the modified dataframe with technical indicators to a new CSV file
    df.to_csv(output_file_path, index=False)
    
# this function will create a new csv file that will put 0-59 row into one row and will add 0 or 1 at the end based upcoming close value
# the order example : 0 - 59 , 1 - 60 , 2 - 61 , 3 - 62 and so on
def write_60rows_on1row(input_file, output_file):
    
    # Get the absolute path of the input CSV file
    input_file_path = get_absolute_path(input_file)
    output_file_path = get_absolute_path(output_file)
    # Reading the input CSV file
    with open(input_file_path, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Creating a new CSV file
    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Writing headers to the new CSV file
        #writer.writerow(data[0])

        # Writing rows to the new CSV file
        for i in tqdm(range(len(data) - 60), desc="Processing rows"):
            # Concatenate the rows into a single list
            concatenated_row = [item for sublist in data[i:i+60] for item in sublist]
            
            # Determine if to append 0 or 1 based on 'close' values
            current_close_value = float(data[i+59][4])  # 'close' value from current set of 60
            upcoming_close_value = float(data[i+60][4])  # 'close' value from upcoming set of 60

            # Append 0 or 1 to the concatenated row
            appended_value = 0 if current_close_value > upcoming_close_value else 1 # if upcoming value is same or bigger -> 1
            concatenated_row.append(appended_value) # add the 0 or 1 at the end of the file
            
            # Writing the concatenated row to the CSV file
            writer.writerow(concatenated_row)
            
if __name__ == '__main__':
    input_csv_file = 'test_BTCUSDT.csv'
    output_csv_file = 'technical_indicators_test_BTCUSDT.csv'
    
    fully_anotate(input_csv_file, output_csv_file)
    