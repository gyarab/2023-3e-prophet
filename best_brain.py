import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt # graphs
from copy import deepcopy as dc

class LSTM(nn.Module):# this class inherits from nn.Module
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        # defines linear function with single ouput neuron 
        self.fc = nn.Linear(hidden_size, 1)
    # function that describes how the data move throgh the model
    def forward(self, x):
        batch_size = x.size(0)
        # Initial hidden and cell states of the LSTM
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # "_" means that we will denote tuple that contains hidden and cell state at the last step
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device
def train_one_epoch(train_loader, epoch, loss_function, optimizer):
    model.train(True)
    print(f"Epoch: {epoch}")
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch) # tensor with 1 value
        # this clears the gradient
        optimizer.zero_grad()
        # gradients are computed for all parameters in nn with respect to the loss
        loss.backward()
        # updates the parameters of the model based on the gradients computed during backpropagation
        optimizer.step()
# train all data
def train_model(train_loader, num_epochs, learning_rate):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_one_epoch(train_loader, epoch, loss_function, optimizer)
        if epoch % 5 == 0:
            save_model()
    save_model()

# Function to save the trained model
def save_model():
    print('saving model')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

# Function to load the trained model
def load_model():
    global model
    print('loading model')
    loaded_model = model #changed
    # It has to be loaded this way if the device does not have cuda device
    if device == 'cpu':
        loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        loaded_model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    model = loaded_model   

# Function to reset models parametres
def reset_model():
    print("reseting models parameters")
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def create_train_graph(X_train, y_train):
    print('creating train graph')
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], look_back + 2))
    dummies[:, 0] = train_predictions
    train_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_train.shape[0], look_back + 2))
    dummies[:, 0] = y_train.flatten()
    new_y_train = dc(dummies[:, 0])
    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', label='Zero Line')
    plt.xlabel('Minute')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()
    
def create_test_graph(X_test, y_test):
    print('creating test graph')
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten() # asi tohle
    dummies = np.zeros((X_test.shape[0], look_back + 2))
    dummies[:, 0] = test_predictions
    test_predictions = dc(dummies[:, 0])
    
    dummies = np.zeros((X_test.shape[0], look_back + 2))
    dummies[:, 0] = y_test.flatten()
    new_y_test = dc(dummies[:, 0])
    plt.plot(new_y_test, label='Actual change')
    plt.plot(test_predictions, label='Predicted change')
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', label='Zero Line')
    plt.xlabel('Minute')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()
    
def create_model_path(model_name = 'not_given'):
    if model_name == 'not_given':
        model_name = f'model_{load_data_mode}_{look_back}LookB_{lstm_neuron_count}neurons_{lstm_layers}L'
        model_name = model_name + '.pth'
    model_path = os.path.join(os.path.dirname(__file__), 'models', model_name)
    
    return model_path
def make_one_prediction(one_sequence_tensor): # to by asi melo byt v best brainu
    one_sequence_tensor = one_sequence_tensor.to(device)
    # Model makes prediction
    with torch.no_grad():
        model.train(False)
        prediction = model(one_sequence_tensor)
        # Here sometimes occures error:
        # a Tensor with 2 elements cannot be converted to Scalar
        # don't know how to solve, so expection setted up
        try:
            prediction_values = prediction.item()
        except:
            print("Prediction ERROR occured:")
            print("a Tensor with 2 elements cannot be converted to Scalar")
            print("Unknown solution")
            print("Setting current prediction to 0.5")
            prediction_values = 0.5
    return prediction_values
#
#Here starts model specific variables
#
device = get_device()
# batch = how muany data points at once will be loaded to the model - increases learning speed, decreases the gpu usage
# after each batch is completed the parameteres of the model will be updated
# if the number of batches is between 1 and the total number of data points in the data set, it is called min-batch gradient descent
# we have: min-batch gradient descent
batch_size = 16 # size of 16 means that 16 datapoints will be loaded at once
look_back = 9 # how many candles will it look into the past
input_file_name = 'not_given'   # this file has to be in /datasets/
# which columns will be included in training data - X
load_data_mode = 2 # modes of loading the data, starts with 0
lstm_layers = 1
lstm_neuron_count = 8
model = LSTM(1, lstm_neuron_count, lstm_layers)
model_path = create_model_path()

    
    
    
    
    
