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
from data_fetcher import Create_price_arr, get_last_100_btc_price
from data_manager import LoaderOHLCV

class LSTM(nn.Module):# this class inherits from nn.Module
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
class TimeSeriesDataset(Dataset):# this class inherits from Dataset
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
def create_batches(train_loader):
    print("creating batches")
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # why break ??
        break
    return x_batch, y_batch

def train_one_epoch(model, train_loader, epoch, loss_function, optimizer):
    model.train(True)
    print()
    print(f'Epoch: {epoch}')
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
def validate_one_epoch(model, test_loader, loss_function):
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
    #print('***************************************************')
    #print()    
    

# train all data
def train_model(model, train_loader, test_loader, num_epochs, model_name):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, epoch, loss_function, optimizer)
        validate_one_epoch(model, test_loader, loss_function)
        if epoch % 10 == 0:
            save_model(model, model_name)

# Function to save the trained model
def save_model(model, model_name):
    model_name = model_name + '.pth'
    print('saving model')
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

# Function to load the trained model
def load_data_model(model, filename):
    print('loading model')
    loaded_model = model #changed
    loaded_model.load_state_dict(torch.load(filename + '.pth'))
    print(f"Model loaded from {filename}")
    return loaded_model   

# Function to reset models parametres
def reset_model(model):
    print("reseting models parameters")
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def create_train_graph(X_train, y_train, look_back, num_of_data_columns, device):
    print('creating train graph')
    scaler = DataManager.get_scaler()
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], look_back * num_of_data_columns + num_of_data_columns + 1)) # !!!!
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_train.shape[0], look_back * num_of_data_columns + num_of_data_columns + 1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', label='Zero Line')
    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()
    
def create_test_graph(X_test, y_test, look_back, num_of_data_columns, device):
    print('creating test graph')
    scaler = DataManager.get_scaler()
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten() # asi tohle
    dummies = np.zeros((X_test.shape[0], look_back * num_of_data_columns + num_of_data_columns + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_test.shape[0], look_back * num_of_data_columns + num_of_data_columns + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', label='Zero Line')
    plt.plot(new_y_test, label='Actual change')
    plt.plot(test_predictions, label='Predicted change')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()




def prepare_live_data(last_prices, look_back, num_of_data_columns):
    # Convert last_prices array to a numpy array
    last_prices_np = np.array(last_prices)
    
    #DEBUG
    print("Columns in last_prices:", last_prices[0])
    # Extract relevant columns
    features_columns = ['Close'
        #,"open", "high", "low", "close", "volume",
        # "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        # "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"
    ]
    
    
    # Get the indices of the features columns
    features_indices = [last_prices[0].index(col) for col in features_columns]

    print("Indices of features columns:", features_indices)

    # Extract features and target values
    X = last_prices_np[:, features_indices]
    target_column_index = last_prices[0].index("Close")
    y = last_prices_np[:, target_column_index]

    # Reshape data for LSTM input
    num_samples = len(last_prices_np)
    X = X.reshape((-1, look_back * num_of_data_columns + num_of_data_columns, 1))
    y = y.reshape((-1, 1))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()

    return X_tensor

def predict_next_value(model, last_prices, look_back, num_of_data_columns, device):
    model.eval()

    # Prepare live data
    X_tensor = prepare_live_data(last_prices, look_back, num_of_data_columns)

    # Make the prediction
    with torch.no_grad():
        prediction = model(X_tensor.to(device)).item()

    print(f'Predicted Next Value: {prediction:.6f}')

    print(f'Predicted Next Value: {prediction:.6f}')

def create_model_name(load_data_mode, features_columns, look_back, lstm_neuron_count,lstm_layers, model_name = 'not_given'):
    if model_name == 'not_given':
        inicials_features_columns = ''.join([s[0] for s in features_columns])
        model_name = f'model_{load_data_mode}_{inicials_features_columns}_{look_back}LookB_{lstm_neuron_count}neurons_{lstm_layers}L'
    
    return model_name
if __name__ == '__main__':
    device = get_device()
    # batch = how muany data points at once will be loaded to the model - increases learning speed, decreases the gpu usage
    # after each batch is completed the parameteres of the model will be updated
    # if the number of batches is between 1 and the total number of data points in the data set, it is called min-batch gradient descent
    # we have: min-batch gradient descent
    batch_size = 16 # size of 16 means that 16 datapoints will be loaded at once
    look_back = 60 # how many candles will it look into the past
    precentage_of_train_data = 0.80 # how much data will be used for training, rest will be used for testing
    input_file_name = 'MinuteBars.csv' # this file has to be in /backend/dataset
    # which columns will be included in training data - X
    features_columns = ['Close',
        #"open", "high", "low", "close", "volume",
        # "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        # "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"
        ]
    num_of_data_columns = len(features_columns) 
    load_data_mode = 2 # modes of loading the data, starts with 0
    lstm_layers = 1
    lstm_neuron_count = 64
    model = LSTM(1, lstm_neuron_count, lstm_layers)
    model.to(device)
    model_name = create_model_name(load_data_mode, features_columns, look_back, lstm_neuron_count, lstm_layers)
    # Train parameters
    learning_rate = 0.001
    num_epochs = 20 # Epoch: Passes the entire training dataset to the model once
    
    if input_file_name == None:
        DataManager = LoaderOHLCV(look_back, features_columns, load_data_mode)
        # Live data preparation
        price_data = Create_price_arr()
        shifted_df_as_np = prepare_live_data(price_data, look_back, num_of_data_columns)
        shifted_df_as_np, scaler = DataManager.scale_data(shifted_df_as_np)  # Assuming scale_data function is correctly defined
        X_tensor, _ = shifted_df_as_np  # Ensure this line correctly unpacks X_tensor
    else:
        # Load the dataset
        DataManager = LoaderOHLCV(look_back, features_columns, load_data_mode, input_file=input_file_name)
        X_train, X_test, y_train, y_test = DataManager.get_data_as_tensor()
        train_dataset, test_dataset = DataManager.to_dataset(X_train, X_test, y_train, y_test)
        train_loader, test_loader = DataManager.to_dataLoader(train_dataset, test_dataset, batch_size)
        train_model(model, train_loader, test_loader, num_epochs, model_name)
        #Load the trained model
        #load_data_model(model, model_name)
        
        # Resets the trained model
        # reset_model(model)
    
    
        # Save the trained model
        save_model(model, model_name) 

    

        #shows graphs
        create_train_graph(X_train, y_train, look_back,num_of_data_columns, device)
        create_test_graph(X_test, y_test, look_back, num_of_data_columns, device)

    
    #x_batch, y_batch = create_batches(train_loader)

    
    
    
    
    
