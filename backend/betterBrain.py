import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame df with columns 'Open', 'High', 'Low', 'Close', 'Volume'
# and 'Close' being the target variable

# Load your data and preprocess
# ...

# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Prepare sequences of 60 OHLCV values for input and corresponding target 'Close' values
X, y = [], []

for i in range(len(df_scaled) - 60):
    X.append(df_scaled[i:i+60])
    y.append(df_scaled[i+60, 3])  # Assuming 'Close' is in the 4th column (index 3)

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the model
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        rnn_out, _ = self.rnn(x)
        out = self.fc(lstm_out[:, -1, :])  # Using only the last timestep's output
        return out

# Instantiate the model
input_size = X_train.shape[2]  # Number of features (OHLCV)
hidden_size = 50  # Number of hidden units
num_layers = 2  # Number of LSTM/RNN layers
output_size = 1  # Output size (single value prediction)

model = HybridModel(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs.squeeze(), y_test)

print(f'Test Loss: {test_loss.item():.4f}')

# Predictions
model.eval()
with torch.no_grad():
    input_data = X_test[:1]  # Taking the first sequence from the test set for prediction
    prediction = model(input_data).item()

# Inverse transform the prediction to get the actual BTC price
predicted_price = scaler.inverse_transform([[0, 0, 0, prediction, 0]])[0, 3]

print(f'Predicted BTC Price: {predicted_price:.2f}')
