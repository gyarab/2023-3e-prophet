import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt # graphs
from copy import deepcopy as dc

# Helper function to get the absolute file path
def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', input_file)
    return input_file_path

file_name = 'technical_indicators_test_BTCUSDT.csv'
data = pd.read_csv(get_absolute_path(file_name))
# show graph
# plt.plot(data['timestamp'], data['close'])
# plt.show()

def prepare_dataframe_for_lstm(dataframe, n_steps):
    # created deepcopy of dataframe - to not edit original data
    dataframe = dc(dataframe)
    
    dataframe.set_index('date', inplace=True)
    
        # Add lag features for the entire OHLCV columns
    features_columns = [
        "open", "high", "low", "close", "volume",
        "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"]

    
    lag_columns = []

    for i in range(1, n_steps + 1):
        for col in features_columns:
            lag_col_name = f'{col}(t-{i})'
            lag_columns.append(dataframe[col].shift(i).rename(lag_col_name))
    
    # Concatenate all lag columns to the original dataframe
    dataframe = pd.concat([dataframe] + lag_columns, axis=1)
    
    # removes possible blank lines
    dataframe.dropna(inplace=True)
    
    return dataframe
# how many candles will it look into the past
look_back = 100
shifted_data_frame = prepare_dataframe_for_lstm(data, look_back)
shifted_df_as_np = shifted_data_frame.to_numpy()

# scales the data
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
