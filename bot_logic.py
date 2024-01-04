import ccxt

#pytorch imports for neural network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_btc_price():
    
    platform = ccxt.bingx()
    # Fetch ticker information for BTC/USDT (you can change the symbol based on your needs)
    ticker = platform.fetch_ticker('BTC/USDT')
    # Extract the last price from the ticker data
    last_price = ticker['last']
    
    return last_price





# Load the dataset
def load_data():
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    X = torch.tensor(dataset[:, 0:8], dtype=torch.float32)
    y = torch.tensor(dataset[:, 8], dtype=torch.float32).reshape(-1, 1)
    return X, y

# Build the neural network model
def build_model():
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 20),
        nn.ReLU(),
        nn.Linear(20, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    return model


# Train the neural network model
def train_model(model, X, y, n_epochs, batch_size):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(n_epochs):
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
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Function to load the trained model
def load_model(model, filename='trained_model.pth'):
    loaded_model = build_model()
    loaded_model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return loaded_model


def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


if __name__ == '__main__':
    #print(get_btc_price())
     
    # Load the dataset
    X, y = load_data()

    # Build the model
    model = build_model()

    # Reset the trained model
    #reset_model(model)


    # Load the trained model
    model = load_model(model)


    # Train the model
    n_epochs = 100 # Epoch: Passes the entire training dataset to the model once
    batch_size = 10 # Batch: One or more samples passed to the model, from which the gradient descent algorithm will be executed for one iteration
    model = train_model(model, X, y, n_epochs, batch_size)

    # Save the trained model
    save_model(model)

    

    # Test the loaded model without retraining
    test_model(model, X, y)
