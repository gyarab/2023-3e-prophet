import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming you have a DataFrame 'df' containing OHLCV and technical indicators data
# Make sure the data is sorted by date in ascending order

# Define the features and target columns
features_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Technical_Indicator_1', 'Technical_Indicator_2']
target_column = 'Close'

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
df[features_columns] = scaler.fit_transform(df[features_columns])
df[target_column] = scaler.fit_transform(df[[target_column]])

# Define a function to create input sequences and labels
def create_sequences_and_labels(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length].values
        label = data.iloc[i+sequence_length][target_column]
        sequences.append(sequence)
        labels.append(label)
    return torch.tensor(sequences), torch.tensor(labels)

# Create sequences and labels
sequence_length = 60
sequences, labels = create_sequences_and_labels(df[features_columns], sequence_length)

# Split the data into training and testing sets
train_size = int(len(sequences) * 0.8)
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Set hyperparameters
input_size = len(features_columns)
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_sequences.float())
    loss = criterion(outputs, train_labels.float())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    test_sequences = test_sequences.float()
    predicted_labels = model(test_sequences)

# Inverse transform the predicted and actual labels
predicted_labels = scaler.inverse_transform(predicted_labels.numpy())
actual_labels = scaler.inverse_transform(test_labels.numpy())

# Calculate and print the Mean Squared Error (MSE)
mse = np.mean((predicted_labels - actual_labels)**2)
print(f'Mean Squared Error: {mse:.4f}')