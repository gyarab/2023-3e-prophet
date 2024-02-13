import os
import pandas as pd
from copy import deepcopy as dc
class LoaderOHLCV():
    def __init__(self, look_back, features_columns,target_column, mode, input_file = 'not_given'):
        self.look_back = look_back
        self.features_columns = features_columns
        self.target_column = target_column
        self.mode = mode
        self.input_file = input_file
        
    def get_data_as_numpy(self):
        if self.input_file == 'not_given':
            pass
        else:
            pass
    def load_data(self):
        print("loading raw data")
        data = pd.read_csv(self.get_absolute_path())
        #different modes of data input
        if self.mode == 0:
            shifted_data_frame = self.prepare_dataframe_for_lstm(data) #'date', 'target_value', 'target_value_difference' + all mentioned columns in features_columns
        if self.mode == 1:
            shifted_data_frame = self.prepare_dataframe_for_lstm2(data) # sequences of returns (differences of values) in featured columns
        if self.mode == 2:
            shifted_data_frame = self.prepare_dataframe_for_lstm3(data) # sequences of returns (differences of values) in featured columns - in %
   
        shifted_df_as_np = shifted_data_frame.to_numpy()
    
        return shifted_df_as_np
    def get_absolute_path(self):
        input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', self.input_file)
        return input_file_path
    
    def prepare_dataframe_for_lstm(self, dataframe):
        # created deepcopy of dataframe - to not edit original data
        selected_columns = ['date']+ self.target_column + self.features_columns
        dataframe = dc(dataframe[selected_columns])
        dataframe.set_index('date', inplace=True) # inplace means it will edit the dataframe
        print(f"shape of loadet data {dataframe.shape}")
        
        # Add lag features for the entire OHLCV columns
        lag_columns = []
        # for i in range(1, n_steps + 1):
        #         for col in features_columns:
        #             dataframe[f'{col}(t-{i})'] = dataframe[col].shift(i)
        
        # adds t-n;look_back (n_steps)> columns of data from previous rows
        for i in range(1, self.look_back + 1):
            for col in self.features_columns:
                lag_col_name = f'{col}(t-{i})'
                lag_columns.append(dataframe[col].shift(i).rename(lag_col_name))
        
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        
        print(f"shape of prepared data{dataframe.shape}")
        return dataframe
    def prepare_dataframe_for_lstm2(self, dataframe):
        selected_columns = ['date'] + self.target_column + self.features_columns # it is expected that just the close value will be passed
        dataframe = dc(dataframe[selected_columns])
        print(f"shape of loadet data {dataframe.shape}")
        
        # Create new DataFrame with 'date' as index
        dataframe.set_index('date', inplace=True) # inplace means it will edit the dataframe
        dataframe['close_difference'] = dataframe['close'].diff()
        
        # adds sequneces of close differences - 1 sequnece will have legnth of n_steps
        lag_columns = []
        for i in range(1, self.look_back + 1):
            lag_col_name = f'close_difference(t-{i})'
            lag_columns.append(dataframe['close_difference'].shift(i).rename(lag_col_name))
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        # removes close column
        dataframe = dataframe.drop('close', axis=1)
        print(f"shape of prepared data {dataframe.shape}")
        #dataframe.to_csv("dataframe_test") # just a debug tool
        return dataframe
    def prepare_dataframe_for_lstm3(self, dataframe):
        # Convert the 'timestamp' column to date if it's not already
        #selected_columns = ['date'] + self.target_column + self.features_columns # it is expected that just the close value will be passed
        dataframe = dc(dataframe[['timestamp', 'close']])
        print(f"shape of loadet data {dataframe.shape}")
        
        # Create new DataFrame with 'date' as index
        dataframe['date'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
        dataframe = dataframe.drop('timestamp', axis=1)
        dataframe.set_index('date', inplace=True) # inplace means it will edit the dataframe
        dataframe['target_vlaue_difference'] = (dataframe['close'].shift(-1) - dataframe['close']) / dataframe['close'] * 100
        dataframe['close_difference'] = (dataframe['close'] - dataframe['close'].shift(1) ) / dataframe['close'].shift(1) * 100
        
        # adds sequneces of close differences - 1 sequnece will have legnth of n_steps
        lag_columns = []
        for i in range(1, self.look_back + 1):
            lag_col_name = f'close_difference(t-{i})'
            lag_columns.append(dataframe['close_difference'].shift(i).rename(lag_col_name))
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        # removes close column
        dataframe = dataframe.drop('close', axis=1)
        print(f"shape of prepared data {dataframe.shape}")
        #dataframe.to_csv("dataframe_test.csv") # just a debug tool
        return dataframe