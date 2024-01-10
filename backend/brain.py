#pytorch imports for neural network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# Function to load the dataset
def load_data():
    print('loading data')
        # Get the current script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the "dataset" folder relative to the script
    dataset_folder = os.path.join(script_dir, '..', 'dataset')

    # Construct the path to the "data" folder within the "dataset" folder
    data_folder = os.path.join(dataset_folder, 'data')

    # Construct the full path to the CSV file
    #csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')
    csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')

    #2251812
    dataset = np.loadtxt(csv_file_path, delimiter=',',skiprows=812, max_rows=500)
    X = torch.tensor(dataset[:, 330:360], dtype=torch.float32)
    y = torch.tensor(dataset[:, 360], dtype=torch.float32).reshape(-1, 1)
    
    
    #dataset = np.loadtxt(csv_file_path, delimiter=',')
    #X = torch.tensor(dataset[:, 0:6], dtype=torch.float32)
    #y = torch.tensor(dataset[:, 6], dtype=torch.float32).reshape(-1, 1)
    
    
    
    return X, y



# Build the neural network model
def build_model():
    print('building model')
    model = nn.Sequential(
        nn.Linear(30, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()# 0 or 1
    )
    return model

# Train the neural network model
def train_model(model, X, y, n_epochs, batch_size):
    print('training model')
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(n_epochs),desc="Processing epochs"):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Finished epoch {epoch}, latest loss {loss}')

    return model

# Function to test the neural network model
def test_model(model, X, y):
    print('testing model')
    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")

    predictions = model(X)
    rounded = predictions.round()

    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), rounded[i].item(), y[i].item()))


# Function to save the trained model
def save_model(model, filename='trained_model.pth'):
    print('saving model')
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Function to load the trained model
def load_model(model, filename='trained_model.pth'):
    print('loading model')
    loaded_model = build_model()
    loaded_model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return loaded_model


def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
if __name__ == '__main__':
 
    # Load the dataset
    X, y = load_data()

    # Build the model
    model = build_model()
    model = model.to(get_device())
    # Reset the trained model
    #reset_model(model)

    # Load the trained model
    #model = load_model(model)


    # Train the model
    n_epochs = 200 # Epoch: Passes the entire training dataset to the model once
    batch_size = 10 # Batch: One or more samples passed to the model, from which the gradient descent algorithm will be executed for one iteration
    
    model = train_model(model, X, y, n_epochs, batch_size)

    # Save the trained model
    save_model(model)

    # Test the loaded model without retraining
    test_model(model, X, y)
