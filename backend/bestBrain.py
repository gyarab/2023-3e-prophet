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

def prepare_dataframe_for_lstm(dataframe, n_steps, features_columns):
    # created deepcopy of dataframe - to not edit original data
    selected_columns = ['date', 'target_value', 'target_value_difference'] + features_columns
    dataframe = dc(dataframe[selected_columns])
    dataframe.set_index('date', inplace=True) # inplace means it will edit the dataframe
    print(f"shape of loadet data {dataframe.shape}")
    
    # Add lag features for the entire OHLCV columns
    lag_columns = []
    # for i in range(1, n_steps + 1):
    #         for col in features_columns:
    #             dataframe[f'{col}(t-{i})'] = dataframe[col].shift(i)
    
    # adds t-<1;look_back (n_steps)> columns of data from previous rows
    for i in range(1, n_steps + 1):
        for col in features_columns:
            lag_col_name = f'{col}(t-{i})'
            lag_columns.append(dataframe[col].shift(i).rename(lag_col_name))
    
    # Concatenate all lag columns to the original dataframe
    dataframe = pd.concat([dataframe] + lag_columns, axis=1)
    
    # removes possible blank lines
    dataframe.dropna(inplace=True)
    
    print(f"shape of prepared data{dataframe.shape}")
    return dataframe
def prepare_dataframe_for_lstm2(dataframe, n_steps, features_columns):
    selected_columns = ['date', 'target_value', 'target_value_difference'] + features_columns # it is expected that just the close value will be passed
    dataframe = dc(dataframe[selected_columns])
    print(f"shape of loadet data {dataframe.shape}")
    
    # Create new DataFrame with 'date' as index
    new_df = dataframe.set_index('date')[['target_value', 'target_value_difference']]

    # Sample DataFrame creation
    # data = {'close': [10, 15, 20, 25, 30]}
    # df = pd.DataFrame(data)

    # # Calculate the difference between upcoming close values
    # df['return'] = df['close'].diff().shift(-1)
 
    #DOES NOT WORK!!!
    # Add columns for the sequence of differences in 'close' values
    for i in range(1, n_steps + 1):
        new_df[f'return_{i}'] = dataframe['close'].shift(i) - dataframe['close'].shift(i + 1)

    # Drop rows with NaN values (due to shift operation)
    new_df = new_df.dropna()

    # removes possible blank lines
    new_df.dropna(inplace=True)
    
    
    print(f"shape of prepared data {new_df.shape}")
    new_df.to_csv("dataframe_test")
    return new_df # just a debug tool
# loads data in acording format
def load_data(file_name, look_back, features_columns, mode):
    print("loading raw data")
    data = pd.read_csv(get_absolute_path(file_name))
    #different modes of data input
    if mode == 0:
        shifted_data_frame = prepare_dataframe_for_lstm(data, look_back, features_columns) #'date', 'target_value', 'target_value_difference' + all mentioned columns in features_columns
    if mode == 1:
        shifted_data_frame = prepare_dataframe_for_lstm2(data, look_back, features_columns) # sequences of returns (differences of values), fea
    
   
    shifted_df_as_np = shifted_data_frame.to_numpy()
    
    return shifted_df_as_np
def absolute_scale_data(shifted_df_as_np):
    shifted_df_as_np = np.where(shifted_df_as_np > 0, 1, -1) # when greater than 0 it will change value to 1 othervise -1
    return shifted_df_as_np
# scales the data
def scale_data(shifted_df_as_np):
    print("scaling data to range -1 .. 1")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    return shifted_df_as_np, scaler

# splits data to training and 
def split_data(shifted_df_as_np, percentage_of_train_data, target_value_index):
    print("spliting data")
    # splits the data into the target value - y and the data based on which y is predicted X
    X = shifted_df_as_np[:, 2:] # upper scale X is correct
    y = shifted_df_as_np[:, target_value_index] # lower scale y - vector ?
    X = dc(np.flip(X, axis=1)) # now the data are in corect time order - older to newer

    split_index = int(len(X) * percentage_of_train_data)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]
    print(f"shape of X_train {X_train.shape}")
    print(f"shape of X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def to_tensor(X_train,X_test,y_train,y_test, look_back, num_of_data_columns):
    print("reshaping data to tensors")
    # reshpaes because LSTM wants 3 dimensional tensors
    X_train = X_train.reshape((-1, look_back * num_of_data_columns + num_of_data_columns , 1)) 
    X_test = X_test.reshape((-1, look_back * num_of_data_columns + num_of_data_columns , 1))

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
    print()
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
    #print('***************************************************')
    #print()    
    

# train all data
def train_model(train_loader, test_loader, num_epochs):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
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

def create_train_graph(X_train, y_train, scaler, look_back, num_of_data_columns, device):
    print('creating train graph')
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], look_back * num_of_data_columns + num_of_data_columns + 2)) # the 2 value is still hard coded
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_train.shape[0], look_back * num_of_data_columns + num_of_data_columns + 2))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])
    
    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()
    
def create_test_graph(X_test, y_test, scaler, look_back, num_of_data_columns, device):
    print('creating test graph')
    
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten() # asi tohle
    dummies = np.zeros((X_test.shape[0], look_back * num_of_data_columns + num_of_data_columns + 2))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_test.shape[0], look_back * num_of_data_columns + num_of_data_columns + 2))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])
    
    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()



def prepare_live_data(last_prices, look_back, num_of_data_columns):
    # Convert last_prices array to a numpy array
    last_prices_np = np.array(last_prices)

    # Extract relevant columns
    features_columns = ['close'
        "open", "high", "low", "close", "volume",
         "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
         "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"
    ]

    # Get the indices of the features columns
    features_indices = [last_prices[0].index(col) for col in features_columns]

    # Extract features and target values
    X = last_prices_np[:, features_indices]
    target_column_index = last_prices[0].index("close")
    y = last_prices_np[:, target_column_index]

    # Reshape data for LSTM input
    num_samples = len(last_prices_np)
    X = X.reshape((-1, look_back * num_of_data_columns + num_of_data_columns, 1))
    y = y.reshape((-1, 1))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()

    return X_tensor, y_tensor

def predict_next_target_difference(model, input_data, device):
    model.eval()

    # Convert input data to tensor
    input_tensor = torch.tensor(input_data).float().to(device)

    # Make the prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()

    print(f'Predicted Next Target Difference: {prediction:.6f}')


if __name__ == '__main__':
    use_dataset = 1


    device = get_device()
    # batch = how muany data points at once will be loaded to the model - increases learning speed, decreases the gpu usage
    # after each batch is completed the parameteres of the model will be updated
    # if the number of batches is between 1 and the total number of data points in the data set, it is called min-batch gradient descent
    # we have: min-batch gradient descent
    batch_size = 16 # size of 16 means that 16 datapoints will be loaded at once
    look_back = 100 # how many candles will it look into the past
    precentage_of_train_data = 0.95 # how much data will be used for training, rest will be used for testing
    file_name = 'technical_indicators_test_BTCUSDT.csv' # this file has to be in /backend/dataset
    # which columns will be included in training data - X
    features_columns = ['close'
        #"open", "high", "low", "close", "volume",
         #"ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
         #"atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"
        ]
    num_of_data_columns = len(features_columns) 
    target_value_index = 1 # what is the target values index (most likely 0 or 1)
    load_mode = 1 # modes of loading the data, starts with 0

    if use_dataset == 1:
        # Load the dataset   
        shifted_df_as_np = load_data(file_name, look_back, features_columns, load_mode)
        shifted_df_as_np, scaler = scale_data(shifted_df_as_np) # scaling is not a good way (the price can get higher than current maximum)
        # shifted_df_as_np = absolute_scale_data(shifted_df_as_np)
    else:
        shifted_df_as_np = prepare_live_data()
        shifted_df_as_np, scaler = scale_data(shifted_df_as_np)
    
    X_train, X_test, y_train, y_test = split_data(shifted_df_as_np, precentage_of_train_data, target_value_index)
    X_train, X_test, y_train, y_test = to_tensor(X_train, X_test, y_train, y_test, look_back, num_of_data_columns)
    train_dataset, test_dataset = to_dataset(X_train, X_test, y_train, y_test)
    train_loader, test_loader = to_dataLoader(train_dataset, test_dataset, batch_size)
    
    #x_batch, y_batch = create_batches(train_loader)
    
    

    # Build the model
    model = LSTM(1, 64, 1)
    model.to(device)
    
    # Load the trained model
    #load_model(model)
    
    # Resets the trained model
    # reset_model(model)


    # Train parameters
    learning_rate = 0.001
    num_epochs = 2 # Epoch: Passes the entire training dataset to the model once
    
    # starts training
    #train_model(train_loader, test_loader, num_epochs)
    
    
    # Save the trained model
    # save_model(model)  #!mozna funguje

   

    #shows graphs
    #create_train_graph(X_train, y_train, scaler, look_back,num_of_data_columns, device)
    #create_test_graph(X_test, y_test, scaler, look_back, num_of_data_columns, device)