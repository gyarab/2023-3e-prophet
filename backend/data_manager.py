import os
import pandas as pd
from copy import deepcopy as dc
class LoaderOHLCV():
    def __init__(self, look_back, features_columns, mode, input_file = 'not_given'):
        self.look_back = look_back
        self.features_columns = features_columns
        self.mode = mode
        self.input_file = input_file
    def get_data_as_numpy(self):
        if self.input_file == 'not_given': # not_given - means live data
            pass
        else:
            shifted_df_as_np = self.load_data_from_csv()
        return shifted_df_as_np
    def load_data_from_csv(self):
        print("loading raw data")
        data = pd.read_csv(self.get_absolute_path())
        #different modes of data input
        if self.mode == 0:
            shifted_data_frame = self.prepare_dataframe_for_lstm0(data) #'Date', 'target_value', 'target_value_difference' + all mentioned columns in features_columns
        if self.mode == 1:
            shifted_data_frame = self.prepare_dataframe_for_lstm1(data) # sequences of returns (differences of values) in featured columns
        if self.mode == 2:
            shifted_data_frame = self.prepare_dataframe_for_lstm2(data) # sequences of returns (differences of values) in featured columns - in %
   
        shifted_df_as_np = shifted_data_frame.to_numpy()
    
        return shifted_df_as_np
    def get_absolute_path(self):
        input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', self.input_file)
        return input_file_path
    # creates sequences of selected data (columns)
    def prepare_dataframe_for_lstm0(self, dataframe):
        # created deepcopy of dataframe - to not edit original data
        try:
            selected_columns = [['Timestamp', 'Close','Open','High','Low','Volume']]
            dataframe = dc(dataframe[selected_columns])
            dataframe['Date'] = pd.to_datetime(dataframe['Timestamp'], unit='ms')
            dataframe = dataframe.drop('Timestamp', axis=1)
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
        except:
            selected_columns = [['Date', 'Close','Open','High','Low','Volume']]
            dataframe = dc(dataframe[selected_columns])
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
        print(f"shape of loadet data {dataframe.shape}")
        
        # Add lag features for the entire OHLCV columns
        lag_columns = []
        
        # adds t-n;look_back (n_steps)> columns of data from previous rows
        for i in range(1, self.look_back + 1):
            for col in self.features_columns:
                lag_col_name = f'{col}(t-{i})'
                lag_columns.append(dataframe[col].shift(i).rename(lag_col_name))
        
        dataframe['Target_value'] = dataframe['Close'].shift(-1)
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        
        print(f"shape of prepared data{dataframe.shape}")
        return dataframe
    
    # creates sequences of difference in Close values
    def prepare_dataframe_for_lstm1(self, dataframe):
        dataframe = dc(dataframe[['Timestamp', 'Close']])
        print(f"shape of loadet data {dataframe.shape}")
        
        # Create new DataFrame with 'Date' as index
        dataframe['Date'] = pd.to_datetime(dataframe['Timestamp'], unit='ms')
        dataframe = dataframe.drop('Timestamp', axis=1)
        dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
        
        dataframe['Close_difference'] = dataframe['Close'].diff()
        dataframe['Target_vlaue_difference'] = dataframe['Close'].shift(-1) - dataframe['Close']
        
        # adds sequneces of Close differences - 1 sequnece will have legnth of n_steps
        lag_columns = []
        for i in range(1, self.look_back + 1):
            lag_col_name = f'Close_difference(t-{i})'
            lag_columns.append(dataframe['Close_difference'].shift(i).rename(lag_col_name))
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        # removes Close column
        dataframe = dataframe.drop('Close', axis=1)
        print(f"shape of prepared data {dataframe.shape}")
        #dataframe.to_csv("dataframe_test") # just a debug tool
        return dataframe
    
    # creates sequences of difference in Close values in percentage
    def prepare_dataframe_for_lstm2(self, dataframe):
        try:
            dataframe = dc(dataframe[['Timestamp', 'Close']])
            dataframe['Date'] = pd.to_datetime(dataframe['Timestamp'], unit='ms')
            dataframe = dataframe.drop('Timestamp', axis=1)
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
        except:
            dataframe = dc(dataframe[['Date', 'Close']])
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
            
        print(f"shape of loadet data {dataframe.shape}")
        
        # creates the target difference columns
        dataframe['Target_vlaue_difference'] = (dataframe['Close'].shift(-1) - dataframe['Close']) / dataframe['Close'] * 100
        dataframe['Close_difference'] = (dataframe['Close'] - dataframe['Close'].shift(1) ) / dataframe['Close'].shift(1) * 100
        
        # adds sequneces of Close differences in percentage - 1 sequnece will have legnth of n_steps
        lag_columns = []
        for i in range(1, self.look_back + 1):
            lag_col_name = f'Close_difference(t-{i})'
            lag_columns.append(dataframe['Close_difference'].shift(i).rename(lag_col_name))
        # Concatenate all lag columns to the original dataframe
        dataframe = pd.concat([dataframe] + lag_columns, axis=1)
        
        # removes possible blank lines
        dataframe.dropna(inplace=True)
        # removes Close column
        dataframe = dataframe.drop('Close', axis=1)
        print(f"shape of prepared data {dataframe.shape}")
        #dataframe.to_csv("dataframe_test.csv") # just a debug tool
        return dataframe