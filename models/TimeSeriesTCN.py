import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3):
        """
        Initialize the Temporal Convolutional Network.

        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of output features.
        kernel_size (int): Size of the convolution kernel.
        """
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output predictions.
        """
        x = self.relu(self.conv1(x))
        return self.conv2(x)

class TimeSeriesTCN:
    def __init__(self, input_size, hidden_size, output_size, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Initialize the TCN training class.

        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of output features.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        """
        self.model = TCN(input_size, hidden_size, output_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, train_data):
        """
        Train the TCN model on the provided data.

        Parameters:
        train_data (array-like): The training time series data.

        Returns:
        None
        """
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(train_data) - self.batch_size):
                x = train_data[i:i + self.batch_size].reshape(self.batch_size, -1, train_data.shape[1])
                y = train_data[i + 1:i + self.batch_size + 1].reshape(self.batch_size, -1)

                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)

                self.optimizer.zero_grad()
                y_pred = self.model(x_tensor)
                loss = self.criterion(y_pred[:, -1, :], y_tensor)
                loss.backward()
                self.optimizer.step()

    def predict(self, input_data, steps):
        """
        Predict future values using the trained model.

        Parameters:
        input_data (array-like): The input data for prediction.
        steps (int): Number of future steps to predict.

        Returns:
        predictions (array): Predicted values.
        """
        self.model.eval()
        input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1, input_data.shape[1])
        predictions = []

        for _ in range(steps):
            with torch.no_grad():
                pred = self.model(input_tensor)
                predictions.append(pred.numpy().flatten())

                # Prepare input for the next time step
                input_tensor = torch.cat((input_tensor[:, 1:, :], pred.reshape(1, 1, -1)), dim=1)

        return np.array(predictions)

    def plot(self, data, predictions):
        """
        Plot the actual data and predictions.

        Parameters:
        data (array-like): Actual time series data.
        predictions (array): Predicted values.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data, label='Actual', color='blue')
        future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
        plt.plot(future_index, predictions, label='Predicted', color='red')
        plt.title("Temporal Convolutional Network Time Series Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)

    # Prepare the data for TCN
    window_size = 10
    input_data = []
    for i in range(len(data) - window_size):
        input_data.append(data.values[i:i + window_size])
    input_data = np.array(input_data)

    # Fit TCN model and forecast
    tcn_model = TimeSeriesTCN(input_size=1, hidden_size=16, output_size=1, epochs=100, batch_size=32)
    tcn_model.train(input_data)
    forecast = tcn_model.predict(input_data[-1:], 30)

    # Plot results
    tcn_model.plot(data, forecast)
