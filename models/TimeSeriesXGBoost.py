import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class TimeSeriesXGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Initialize the XGBoost model for time series forecasting.
        
        Parameters:
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage.
        max_depth (int): Maximum depth of the trees.
        random_state (int): Random state for reproducibility.
        """
        self.model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                   max_depth=max_depth, random_state=random_state)

    def create_dataset(self, data, seq_length):
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

    def train(self, data, seq_length):
        """
        Train the XGBoost model on the training data.
        
        Parameters:
        data (array): The time series data.
        seq_length (int): The length of the input sequences.
        
        Returns:
        None
        """
        X, y = self.create_dataset(data, seq_length)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        # Calculate and print the Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.4f}')

    def predict(self, data, seq_length, steps=1):
        """
        Predict future values using the trained model.
        
        Parameters:
        data (array): Input data for prediction.
        seq_length (int): The length of the input sequences.
        steps (int): Number of future steps to predict.
        
        Returns:
        predictions (array): Predicted values.
        """
        predictions = []
        current_input = data[-seq_length:].tolist()

        for _ in range(steps):
            # Create the next input sequence for prediction
            next_input = np.array(current_input[-seq_length:]).reshape(1, -1)
            next_pred = self.model.predict(next_input)
            predictions.append(next_pred[0])

            # Update the current input
            current_input.append(next_pred[0])
        
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500)

    # Normalize the data (optional)
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Create and train the model
    model = TimeSeriesXGBoost(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.train(data_normalized, seq_length=10)

    # Predict future values
    predictions = model.predict(data_normalized, seq_length=10, steps=10)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time[10:], data_normalized[10:], label='Actual', color='blue')
    plt.plot(time[-10:], predictions, label='Predicted', color='red')
    plt.title("XGBoost Time Series Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()
