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
    input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', input_file)
    return input_file_path

# Function to load data from a CSV file
def load_data(csv_file):
    # Get the absolute file path using a helper function
    file_path = get_absolute_path(csv_file)
    # Print a message indicating data loading
    print('Loading data')
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    # Save the 'timestamp' column for later use
    timestamps = df['timestamp']
    
    # Define the columns for features and the target
    # It excludes the timestamp from normalization
    features_columns = [
        "open", "high", "low", "close", "volume",
        "ema_14", "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        "atr", "ichimoku_a", "ichimoku_b", "obv", "williams_r", "adx"]

    target_column = 'target_value'

    # Use MinMaxScaler to normalize the feature and target columns
    scaler = MinMaxScaler()
    df[features_columns + [target_column]] = scaler.fit_transform(df[features_columns + [target_column]])

    # Create sequences and labels for the LSTM model
    sequence_length = 60
    sequences, labels = create_sequences_and_labels(df[features_columns + [target_column]], sequence_length) #includes target_value_column
    
    # Split the data into training and testing sets
    # -- randomization needs to be added--
    train_size = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    # Return the training and testing data along with timestamps
    return train_sequences, train_labels, test_sequences, test_labels, timestamps

def create_sequences_and_labels(data, sequence_length):
    # Initialize empty lists to store sequences and labels
    sequences, labels = [], []
    
    # Iterate through the data to create sequences and labels
    for i in range(len(data) - sequence_length):
        # Extract a sequence of length 'sequence_length' from the data
        sequence = data[i:i+sequence_length].values
        
        # Extract the target value (label) for the current sequence
        label = data.iloc[i+sequence_length]['target_value']
        
        # Append the sequence and label to their respective lists
        sequences.append(sequence)
        labels.append(label)
    
    # Convert lists to NumPy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Convert NumPy arrays to a single PyTorch tensor
    sequences = torch.tensor(sequences, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Return the sequences and labels as PyTorch tensors
    return sequences, labels

# Function to build the LSTM model
def build_model(input_size, hidden_size, num_layers, output_size, sequence_length):
    print('Building model')
    model = nn.Sequential(
        nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),      
        nn.Linear(hidden_size * sequence_length, output_size)
            )

    
    return model

# Function to train the LSTM model
def train_model(model, X, y, n_epochs, batch_size, learning_rate):
    print('Training model')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device
    device = get_device()
    model = model.to(device)

    # Convert X and y to PyTorch tensors and move them to the device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

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
    file_path = 'technical_indicators_test_BTCUSDT.csv'
    train_sequences, train_labels, test_sequences, test_labels, timestamps = load_data(file_path)

    # Define the input size based on the shape of the first sequence in the training data
    input_size = len(train_sequences[0][0])
    # Define the size of the hidden layer in the LSTM model
    hidden_size = 50
    # Specify the number of layers in the LSTM model
    num_layers = 2
    # Define the output size of the model
    output_size = 1
    # Define the sequnece lenght of each data, how many candles in the past
    sequence_length = 60
    # Set the number of epochs for training
    n_epochs = 100
    # Specify the batch size for training
    batch_size = 10
    # Set the learning rate for the optimizer
    learning_rate = 0.001


    model = build_model(input_size, hidden_size, num_layers, output_size, sequence_length)
    model = model.to(get_device())

    # Create a new MinMaxScaler and fit it only on the training data
    scaler = MinMaxScaler()

    # Exclude timestamp column from normalization
    train_sequences = scaler.fit_transform(train_sequences.reshape(-1, input_size)).reshape(train_sequences.shape)
    train_labels = scaler.fit_transform(train_labels.reshape(-1, 1)).reshape(train_labels.shape)

    model = train_model(model, train_sequences, train_labels, n_epochs, batch_size, learning_rate)

    save_model(model)

    # Use the same scaler to transform the test data
    test_sequences_scaled = scaler.transform(test_sequences.reshape(-1, input_size)).reshape(test_sequences.shape)
    test_labels_scaled = scaler.transform(test_labels.reshape(-1, 1)).reshape(test_labels.shape)

    test_model(model, test_sequences_scaled, test_labels_scaled, scaler)