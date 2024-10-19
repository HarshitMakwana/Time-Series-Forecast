import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the LSTM model.
        
        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of LSTM units in each layer.
        num_layers (int): The number of LSTM layers.
        output_size (int): The number of output features.
        """
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Parameters:
        x (Tensor): Input tensor.
        
        Returns:
        Tensor: Output predictions.
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

class TimeSeriesForecasting:
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001, num_epochs=100):
        """
        Initialize the Time Series Forecasting model.
        
        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of LSTM units in each layer.
        num_layers (int): The number of LSTM layers.
        output_size (int): The number of output features.
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        """
        self.model = LSTMTimeSeries(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def train(self, train_data, train_labels):
        """
        Train the LSTM model on the training data.
        
        Parameters:
        train_data (Tensor): Training input data.
        train_labels (Tensor): Training target data.
        
        Returns:
        None
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.criterion(output, train_labels)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, data):
        """
        Predict future values using the trained model.
        
        Parameters:
        data (Tensor): Input data for prediction.
        
        Returns:
        predictions (Tensor): Predicted values.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

def create_dataset(data, seq_length):
    """
    Create sequences and corresponding labels for training.
    
    Parameters:
    data (array): The time series data.
    seq_length (int): The length of the input sequences.
    
    Returns:
    X (array): Input sequences.
    y (array): Corresponding labels.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Example usage:
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Create sequences
    seq_length = 10
    X, y = create_dataset(data_normalized, seq_length)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).view(-1, seq_length, 1)  # Reshape to [batch_size, seq_length, input_dim]
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Create and train the model
    model = TimeSeriesForecasting(input_size=1, hidden_size=64, num_layers=2, output_size=1, lr=0.001, num_epochs=100)
    model.train(X_tensor, y_tensor)

    # Predict future values
    predictions = model.predict(X_tensor[-10:])  # Use the last 10 data points to predict the next value

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time[seq_length:], data_normalized[seq_length:], label='Actual', color='blue')
    plt.plot(time[-10:], predictions.numpy(), label='Predicted', color='red')
    plt.title("LSTM Time Series Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()

    # Evaluate the forecast
    mse = mean_squared_error(y_tensor.numpy()[-10:], predictions.numpy())
    print(f'Mean Squared Error: {mse:.4f}')
