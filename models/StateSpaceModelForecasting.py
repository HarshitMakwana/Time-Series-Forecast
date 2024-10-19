import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

class StateSpaceModelForecasting:
    def __init__(self):
        """
        Initialize the State Space Model for forecasting.
        """
        self.model = None
        self.results = None

    def fit(self, data, order, seasonal_order=None):
        """
        Fit a State Space Model (SARIMAX) to the time series data.
        
        Parameters:
        data (Series): The input time series data.
        order (tuple): The (p, d, q) order of the model.
        seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.
        
        Returns:
        None
        """
        self.model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        self.results = self.model.fit(disp=False)

    def forecast(self, steps=1):
        """
        Forecast future values using the fitted State Space model.
        
        Parameters:
        steps (int): The number of future periods to predict.
        
        Returns:
        forecast (Series): The forecasted values.
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        forecast = self.results.get_forecast(steps=steps)
        return forecast.predicted_mean

    def plot_forecast(self, data, forecast):
        """
        Plot the actual values and forecasted values.
        
        Parameters:
        data (Series): The actual time series data.
        forecast (Series): The forecasted values.
        
        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data, label='Actual', color='blue')
        plt.plot(forecast.index, forecast, label='Forecast', color='red')
        plt.title("Actual vs Forecasted values")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def evaluate(self, actual_data, forecast):
        """
        Evaluate the forecast by calculating the Mean Squared Error (MSE).
        
        Parameters:
        actual_data (Series): The actual time series data for the evaluation.
        forecast (Series): The forecasted values.
        
        Returns:
        mse (float): The mean squared error between the actual and forecasted values.
        """
        mse = mean_squared_error(actual_data, forecast)
        print(f'Mean Squared Error: {mse}')
        return mse

# Example usage:
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)
    train_data = data[:400]
    test_data = data[400:]

    # Create State Space Model forecasting object
    model = StateSpaceModelForecasting()

    # Fit the model (order can be adjusted based on data)
    model.fit(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 50))

    # Forecast future values
    forecast_values = model.forecast(steps=len(test_data))

    # Plot the forecast
    model.plot_forecast(test_data, forecast_values)

    # Evaluate the forecast
    model.evaluate(test_data, forecast_values)
