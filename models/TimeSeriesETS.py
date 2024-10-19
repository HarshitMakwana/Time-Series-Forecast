import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

class TimeSeriesETS:
    def __init__(self, seasonal='add', seasonal_periods=None):
        """
        Initialize the ETS model for time series forecasting.
        
        Parameters:
        seasonal (str): Type of seasonal component ('add', 'mul', or None).
        seasonal_periods (int): Number of periods in each season (e.g., 12 for monthly data).
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def train(self, data):
        """
        Train the ETS model on the provided data.
        
        Parameters:
        data (array-like): The time series data.
        
        Returns:
        None
        """
        self.model = ExponentialSmoothing(data, seasonal=self.seasonal, 
                                           seasonal_periods=self.seasonal_periods).fit()

    def predict(self, steps):
        """
        Predict future values using the trained model.
        
        Parameters:
        steps (int): Number of future steps to predict.
        
        Returns:
        predictions (array): Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.forecast(steps)

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
        plt.plot(predictions.index, predictions, label='Predicted', color='red')
        plt.title("ETS Time Series Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)

    # Create and train the model
    model = TimeSeriesETS(seasonal='add', seasonal_periods=50)
    model.train(data)

    # Predict future values
    steps = 10
    predictions = model.predict(steps)

    # Create a date range for predictions
    prediction_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)
    predictions = pd.Series(predictions, index=prediction_index)

    # Plot the results
    model.plot(data, predictions)
