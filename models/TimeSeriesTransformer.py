import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, n_layers, output_dim):
        """
        Initialize the Transformer model.
        
        Parameters:
        input_dim (int): The dimension of the input features.
        model_dim (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of layers in the Transformer.
        output_dim (int): The dimension of the output.
        """
        super(TransformerTimeSeries, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, n_heads),
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

class TimeSeriesForecasting:
    def __init__(self, input_dim, model_dim, n_heads, n_layers, output_dim, lr=0.001, num_epochs=100):
        """
        Initialize the Time Series Forecasting model.
        
        Parameters:
        input_dim (int): The dimension of the input features.
        model_dim (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of layers in the Transformer.
        output_dim (int): The dimension of the output.
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        """
        self.model = TransformerTimeSeries(input_dim, model_dim, n_heads, n_layers, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def train(self, train_data, train_labels):
        """
        Train the Transformer model on the training data.
        
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
            loss = self.criterion(output.view(-1), train_labels.view(-1))
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
    y_tensor = torch.FloatTensor(y)

    # Create and train the model
    model = TimeSeriesForecasting(input_dim=1, model_dim=64, n_heads=4, n_layers=2, output_dim=1, lr=0.001, num_epochs=100)
    model.train(X_tensor, y_tensor)

    # Predict future values
    predictions = model.predict(X_tensor[-10:])  # Use the last 10 data points to predict the next value

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time[seq_length:], data_normalized[seq_length:], label='Actual', color='blue')
    plt.plot(time[-10:], predictions.numpy(), label='Predicted', color='red')
    plt.title("Transformer Time Series Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()

    # Evaluate the forecast
    mse = mean_squared_error(y_tensor.numpy()[-10:], predictions.numpy())
    print(f'Mean Squared Error: {mse:.4f}')
