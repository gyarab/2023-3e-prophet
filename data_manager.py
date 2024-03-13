import os
import pandas as pd
from copy import deepcopy as dc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import Create_price_arr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# class for creating dataset
class TimeSeriesDataset(Dataset):# this class inherits from Dataset
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LoaderOHLCV():
    def __init__(self, look_back, features_columns, mode, input_file = 'not_given'):
        self.look_back = look_back
        self.features_columns = features_columns
        self.mode = mode
        self.input_file = input_file
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def get_data_as_tensor(self):
        if self.input_file == 'not_given':
            
            raw_data = Create_price_arr()
            #Hard coded mode of prepare_dataframe_for_lstm - seted to 2
            #FIX!!
            shifted_df_as_np = self.X_train, X_test, y_train, y_test(pd.DataFrame(raw_data))
            shifted_df_as_np = self.scale_data(shifted_df_as_np)
            #FIX!!
            return #chybi aby to byl tensor
        # this is case where there is given file to load from
        # it returns tensors with y and X values and also splits them into to test and train
        else:
            shifted_df_as_np = self.load_data_from_csv()
            shifted_df_as_np = self.scale_data(shifted_df_as_np)
            X_train, X_test, y_train, y_test = self.split_data(shifted_df_as_np)
            X_train, X_test, y_train, y_test = self.to_train_tensor(X_train, X_test, y_train, y_test)
            return X_train, X_test, y_train, y_test    
    
    def split_data(self, shifted_df_as_np, percentage_of_train_data = 0.8):
        print("spliting data")
        # splits the data into the target value - y and the data based on which y is predicted X
        X = shifted_df_as_np[:, 1:] # upper scale X is correct
        y = shifted_df_as_np[:, 0] # lower scale y - vector ?
        X = dc(np.flip(X, axis=1)) # now the data are in corect time order - older to newer

        split_index = int(len(X) * percentage_of_train_data)

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]
        print(f"shape of X_train {X_train.shape}")
        print(f"shape of X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    def to_train_tensor(self, X_train, X_test, y_train, y_test):
        print("reshaping data to tensors")
        # reshpaes because LSTM wants 3 dimensional tensors
        #HARD CODED TO X_train, X_test, y_train, y_test - it has 1 column of data 
        #FIX!!
        X_train = X_train.reshape((-1, self.look_back * 1 + 1 , 1)) 
        X_test = X_test.reshape((-1, self.look_back * 1 + 1 , 1))

        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))
        # moves to tensor
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()
        
        return X_train, X_test, y_train, y_test
    def to_dataset(self, X_train, X_test, y_train, y_test):
        print("to_dataset")
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        return train_dataset, test_dataset
    def to_dataLoader(self, train_dataset, test_dataset, batch_size):
        print("to_dataloader")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader    
        
    
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
        try:
            dataframe = dc(dataframe[['Timestamp', 'Close']])
            dataframe['Date'] = pd.to_datetime(dataframe['Timestamp'], unit='ms')
            dataframe = dataframe.drop('Timestamp', axis=1)
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
        except:
            dataframe = dc(dataframe[['Date', 'Close']])
            dataframe.set_index('Date', inplace=True) # inplace means it will edit the dataframe
            
        print(f"shape of loadet data {dataframe.shape}")
    
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
    def absolute_scale_data(self, shifted_df_as_np):
        shifted_df_as_np = np.where(shifted_df_as_np > 0, 1, -1) # when greater than 0 it will change value to 1 othervise -1
        return shifted_df_as_np
    # scales the data
    def scale_data(self, shifted_df_as_np):
        print("scaling data to range -1 .. 1")
        shifted_df_as_np = self.scaler.fit_transform(shifted_df_as_np)
        return shifted_df_as_np
    # returns whole path to the dataset/data folder
    def get_absolute_path(self):
        input_file_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', self.input_file)
        return input_file_path
    def get_scaler(self):
        return self.scaler