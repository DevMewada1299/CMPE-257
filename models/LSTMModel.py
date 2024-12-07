import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
import torch.nn
from sklearn.preprocessing import MinMaxScaler

df = yf.download('NVDA', start='2024-01-01', end='2024-12-01')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # First LSTM layer with Dropout
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second LSTM layer with Dropout
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Third LSTM layer with Dropout
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.dropout3 = nn.Dropout(p=0.4)
        
        # Fourth LSTM layer with Dropout
        self.lstm4 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.dropout4 = nn.Dropout(p=0.5)
        
        # Fully connected output layer
        self.fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # Pass through first LSTM layer and apply dropout
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Pass through second LSTM layer and apply dropout
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Pass through third LSTM layer and apply dropout
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # Pass through fourth LSTM layer and apply dropout
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        
        # Take the output of the last time step and pass it through the fully connected layer
        x = self.fc(x[:, -1, :])

        return x
    
model = Model()

scalers = {
    'Adj Close': MinMaxScaler(),
    'Close': MinMaxScaler(),
    'Low': MinMaxScaler(),
    'High': MinMaxScaler(),
    'Open': MinMaxScaler(),
    'Volume' : MinMaxScaler()
}

df['Adj Close'] = scalers['Adj Close'].fit_transform(df[['Adj Close']])
df['Close'] = scalers['Close'].fit_transform(df[['Close']])
df['Low'] = scalers['Low'].fit_transform(df[['Low']])
df['High'] = scalers['High'].fit_transform(df[['High']])
df['Open'] = scalers['Open'].fit_transform(df[['Open']])
df['Volume'] = scalers['Volume'].fit_transform(df[['Volume']])

df = df.drop(columns=['Adj Close'])
df_data = df.values



# Convert data into sequences
def create_sequences(data_array, sequence_length):
    X = []
    y = []
    for i in range(len(data_array) - sequence_length):
        # Extract sequence of features
        X.append(data_array[i:i + sequence_length])
        # Target is the closing price at the next timestep
        y.append(data_array[i + sequence_length][0])  
    return np.array(X), np.array(y)

sequence_length = 5

X_sequences, y_targets = create_sequences(df_data, sequence_length)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_targets, train_size=0.8, shuffle=False)

X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
y_tensor_train = torch.tensor(y_train, dtype=torch.float32)
X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
y_tensor_test = torch.tensor(y_test, dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset
dataset_train = TensorDataset(X_tensor_train, y_tensor_train)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0
    
    for X_batch, y_batch in dataloader_train:
        optimizer.zero_grad()  # Clear gradients from the previous step
        
        y_pred = model(X_batch)  # Forward pass
        
        loss = criterion(y_pred.squeeze(), y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item() * X_batch.size(0)  # Accumulate loss
    
    epoch_loss /= len(dataloader_train.dataset)  # Average loss over all samples
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

model.eval()  # Set model to evaluation mode
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in dataloader_test:
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        test_loss += loss.item() * X_batch.size(0)

test_loss /= len(dataloader_test.dataset)  # Average test loss

print(f"Test Loss: {test_loss:.4f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predictions on test set
y_pred_list = []
y_true_list = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in dataloader_test:
        y_pred_list.append(model(X_batch).squeeze().numpy())
        y_true_list.append(y_batch.numpy())

y_pred_list = np.concatenate(y_pred_list)
y_true_list = np.concatenate(y_true_list)

mse = mean_squared_error(y_true_list, y_pred_list)
mae = mean_absolute_error(y_true_list, y_pred_list)

print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")


torch.save(model.state_dict(), "models/lstm_model.pth")


input_data = torch.rand(1,1,5) #Your Input to the model, please include this in 

#loading the model
model = Model()
model.load_state_dict(torch.load("models/lstm_model.pth"))

model.eval()
with torch.no_grad():  
    output = model(input_data)

print(output)



def future_predictions(X):
    current_input = X[-1].unsqueeze(0) 
    future_predictions = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            # Predict the next value
            next_pred = model(current_input)  # Shape: (1, 1)
    
            # Store the prediction
            future_predictions.append(next_pred.item())
    
            # Update the input sequence
            next_input = current_input[:, 1:, :]  # Remove the first timestep (shift the window)
            next_input = torch.cat((next_input, torch.zeros(1, 1, current_input.size(2))), dim=1)  # Add placeholder for the new timestep
    
            # Insert the predicted value into the second column (target feature)
            next_input[:, -1, 1] = next_pred.squeeze(1)  # Replace column index 1 with prediction
    
            current_input = next_input

    return future_predictions

future_preds = future_predictions(input_data)

print(scalers['Close'].inverse_transform(np.array(future_preds).reshape(-1,1))) #Final preds for next 5 days.
