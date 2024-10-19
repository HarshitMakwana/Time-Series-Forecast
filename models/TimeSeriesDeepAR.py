import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class DeepAR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the DeepAR model.

        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of output features.
        """
        super(DeepAR, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output predictions.
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TimeSeriesDeepAR:
    def __init__(self, input_size, hidden_size, output_size, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Initialize the DeepAR training class.

        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of output features.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        """
        self.model = DeepAR(input_size, hidden_size, output_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, train_data):
        """
        Train the DeepAR model on the provided data.

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
                loss = self.criterion(y_pred, y_tensor)
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
        predictions = []
        input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1, input_data.shape[1])

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
        plt.title("DeepAR Time Series Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)

    # Data preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Create and train the model
    model = TimeSeriesDeepAR(input_size=1, hidden_size=64, output_size=1, epochs=100, batch_size=32)
    model.train(scaled_data)

    # Prepare input for prediction
    input_data = scaled_data[-32:]  # Use the last 32 time steps for prediction
    steps = 10
    predictions = model.predict(input_data, steps)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)

    # Plot the results
    model.plot(data, predictions)
