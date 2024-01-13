import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# Helper function to get the absolute file path
def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset', input_file)
    return input_file_path

# Function to load data from a CSV file
def load_data(csv_file):
    # Get the absolute file path using a helper function
    file_path = get_absolute_path(csv_file)
    # Print a message indicating data loading
    print('Loading data')
    
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Define the columns for features and the target
    features_columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'EMA_14', 'RSI_14', 'MACD','BollingerUpper', 'BollingerLower',
                        'ATR', 'IchimokuA', 'IchimokuB', 'OBV', 'WilliamsR', 'ADX']
    target_column = 'TargetValue'

    # Use MinMaxScaler to normalize the feature and target columns
    scaler = MinMaxScaler()
    df[features_columns] = scaler.fit_transform(df[features_columns])
    df[target_column] = scaler.fit_transform(df[[target_column]])

    # Create sequences and labels for the LSTM model
    sequence_length = 60
    sequences, labels = create_sequences_and_labels(df[features_columns], sequence_length)

    # Split the data into training and testing sets
    # -- randomization needs to be added--
    train_size = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    # Return the training and testing data
    return train_sequences, train_labels, test_sequences, test_labels

# Function to create sequences and labels
def create_sequences_and_labels(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length].values
        label = data.iloc[i+sequence_length]['Close']
        sequences.append(sequence)
        labels.append(label)
    return torch.tensor(sequences), torch.tensor(labels)

# Function to build the LSTM model
def build_model(input_size, hidden_size, num_layers, output_size):
    print('Building model')
    model = nn.Sequential(
        nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
        nn.Linear(hidden_size, output_size)
    )
    return model

# Function to train the LSTM model
def train_model(model, X, y, n_epochs, batch_size, learning_rate):
    print('Training model')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert X and y to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Calculate the number of batches
    num_batches = len(X_tensor) // batch_size

    for epoch in tqdm(range(n_epochs), desc="Processing epochs"):
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size

            # Extract the current batch
            X_batch = X_tensor[start_idx:end_idx]
            y_batch = y_tensor[start_idx:end_idx]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model


# Function to test the trained LSTM model
def test_model(model, X, y, scaler):
    print('Testing model')
    model.eval()
    with torch.no_grad():
        predicted_labels = model(X.float())

    # Inverse transform the predicted and actual labels
    predicted_labels = scaler.inverse_transform(predicted_labels.numpy())
    actual_labels = scaler.inverse_transform(y.numpy())

    # Calculate and print the Mean Squared Error (MSE)
    mse = np.mean((predicted_labels - actual_labels)**2)
    print(f'Mean Squared Error: {mse:.4f}')

# Function to save the trained model
def save_model(model, filename='trained_model.pth'):
    print('Saving model')
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Function to get the available computing device (CPU or GPU)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main part of the script
if __name__ == '__main__':
    file_path = 'your_data.csv'
    train_sequences, train_labels, test_sequences, test_labels = load_data(file_path)

    input_size = len(train_sequences[0][0])
    hidden_size = 50
    num_layers = 2
    output_size = 1
    n_epochs = 100
    batch_size = 10
    learning_rate = 0.001

    model = build_model(input_size, hidden_size, num_layers, output_size)
    model = model.to(get_device())

    # Create a new MinMaxScaler and fit it only on the training data
    scaler = MinMaxScaler()
    train_sequences = scaler.fit_transform(train_sequences.reshape(-1, input_size)).reshape(train_sequences.shape)
    train_labels = scaler.fit_transform(train_labels.reshape(-1, 1)).reshape(train_labels.shape)

    model = train_model(model, train_sequences, train_labels, n_epochs, batch_size, learning_rate)

    save_model(model)

    # Use the same scaler to transform the test data
    test_sequences_scaled = scaler.transform(test_sequences.reshape(-1, input_size)).reshape(test_sequences.shape)
    test_labels_scaled = scaler.transform(test_labels.reshape(-1, 1)).reshape(test_labels.shape)

    test_model(model, test_sequences_scaled, test_labels_scaled, scaler)
