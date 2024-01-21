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
    
# class for creating dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device
# Helper function to get the absolute file path
def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', input_file)
    return input_file_path

def prepare_dataframe_for_lstm(dataframe, n_steps):
    # created deepcopy of dataframe - to not edit original data
    dataframe = dc(dataframe)
    
    dataframe.set_index('date', inplace=True)
    print(dataframe.shape)
    # Add lag features for the entire OHLCV columns
    features_columns = [
        "open", "high", "low", "close", "volume",
        "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"]

    
    lag_columns = []
    # for i in range(1, n_steps + 1):
    #         for col in features_columns:
    #             dataframe[f'{col}(t-{i})'] = dataframe[col].shift(i)

    for i in range(1, n_steps + 1):
        for col in features_columns:
            lag_col_name = f'{col}(t-{i})'
            lag_columns.append(dataframe[col].shift(i).rename(lag_col_name))
    
    # Concatenate all lag columns to the original dataframe
    dataframe = pd.concat([dataframe] + lag_columns, axis=1)
    
    # removes possible blank lines
    dataframe.dropna(inplace=True)
    
    print(dataframe.shape)
    return dataframe
# loads data in acording format
def load_data(file_name, look_back):
    print("loading raw data")
    data = pd.read_csv(get_absolute_path(file_name))
    shifted_data_frame = prepare_dataframe_for_lstm(data, look_back)
    # shifted_data_frame.to_csv("test.csv")
    shifted_df_as_np = shifted_data_frame.to_numpy()
    
    return shifted_df_as_np

# scales the data
def scale_data(shifted_df_as_np):
    print("scaling data to range -1 .. 1")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    return shifted_df_as_np, scaler

# splits data to training and 
def split_data(shifted_df_as_np, percentage_of_train_data):
    print("spliting data")
    # splits the data into the target value - y and the data based on which y is predicted X
    X = shifted_df_as_np[:, 2:] # upper scale X is correct
    y = shifted_df_as_np[:, 0] # lower scale y - vector ?
    X = dc(np.flip(X, axis=1)) # now the data are in corect time order - older to newer

    split_index = int(len(X) * percentage_of_train_data)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def to_tensor(X_train,X_test,y_train,y_test):
    print("reshaping data to tensors")
    # reshpaes because LSTM wants 3 dimensional tensors
    X_train = X_train.reshape((-1, 100 * 16 + 16 , 1)) # hard coded!!!!, it is not responsive to lenght of data!!
    X_test = X_test.reshape((-1, 100 * 16 + 16, 1)) # hard coded!!!!, it is not responsive to lenght of data!!

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # moves to tensor
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    return X_train,X_test,y_train,y_test

def to_dataset(X_train,X_test,y_train,y_test):
    print("to_dataset")
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    return train_dataset, test_dataset

def to_dataLoader(train_dataset,test_dataset,batch_size):
    print("to_dataloader")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_batches(train_loader):
    print("creating batches")
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # why break ??
        break
    return x_batch, y_batch

def train_one_epoch(train_loader, epoch, loss_function, optimizer):
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
def validate_one_epoch(test_loader, loss_function):
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
    
    

# train all data
def train_model(train_loader,test_loader, num_epochs):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs), desc="Processing epochs"):
        train_one_epoch(train_loader, epoch, loss_function, optimizer)
        validate_one_epoch(test_loader, loss_function)

# Function to save the trained model
def save_model(model, filename='trained_model.pth'):
    print('saving model')
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Function to load the trained model
def load_model(model, filename='trained_model.pth'):
    print('loading model')
    loaded_model = model #changed
    loaded_model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return loaded_model   

# Function to reset models parametres
def reset_model(model):
    print("reseting models parameters")
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
def create_graph(X_train, y_train, scaler, lookback, device):
    print('creating graph')
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], lookback * 16 + 16 + 2))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_train.shape[0], lookback * 16 + 16 + 2))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])
    
    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    device = get_device()
    
    # k cemu je ?
    batch_size = 16 # Batch: One or more samples passed to the model, from which the gradient descent algorithm will be executed for one iteration

    
    # Load the dataset   
    look_back = 100 # how many candles will it look into the past
    shifted_df_as_np = load_data('technical_indicators_test_BTCUSDT.csv', look_back)
    shifted_df_as_np, scaler = scale_data(shifted_df_as_np)
    X_train, X_test, y_train, y_test = split_data(shifted_df_as_np, 0.80)
    X_train, X_test, y_train, y_test = to_tensor(X_train, X_test, y_train, y_test)
    train_dataset, test_dataset = to_dataset(X_train, X_test, y_train, y_test)
    train_loader, test_loader = to_dataLoader(train_dataset, test_dataset, batch_size)
    # x_batch, y_batch = create_batches(train_loader)
    
    # Build the model
    model = LSTM(1, 4, 1)
    model.to(device)
    
    
    # Reset the trained model
    # reset_model(model)   #!mozna funguje

    # Load the trained model
    load_model(model)   #!mozna funguje


    # Train the model
    learning_rate = 0.001
    num_epochs = 10 # Epoch: Passes the entire training dataset to the model once
    
    # starts training
    # train_model(train_loader, test_loader, num_epochs)
    create_graph(X_train, y_train, scaler, look_back, device)
    # Save the trained model
    # save_model(model)  #!mozna funguje

    # Test the loaded model without retraining
    #test_model(model, X, y)    