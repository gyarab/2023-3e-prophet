import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt # graphs
from copy import deepcopy as dc

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
# splits the data into the target value - y and the data based on whic y is predicted X
X = shifted_df_as_np[:, 1:] # upper scale is correct
y = shifted_df_as_np[:, 0] # lower scale - vector ?

X = dc(np.flip(X, axis=1)) # now the data are in corect time order - older to newer

split_index = int(len(X) * 0.95)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# reshpaes because LSTM wants 3 dimensional tensors
X_train = X_train.reshape((-1, look_back, 1))
X_test = X_test.reshape((-1, look_back, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
# moves to tensor
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# class for creating dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)


batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    # why ?
    break

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # hodne skifi, asi neresit
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch) # tensor with 1 value
        running_loss += loss.item() # gets the 1 value
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()    
    
    
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs), desc="Processing epochs"):
    train_one_epoch()
    validate_one_epoch()