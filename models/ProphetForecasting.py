import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

class ProphetForecasting:
    def __init__(self):
        """
        Initialize the Prophet model for forecasting.
        """
        self.model = None
        self.history = None

    def prepare_data(self, data):
        """
        Prepare the data for the Prophet model.
        
        Parameters:
        data (DataFrame): The input time series data with two columns: 'ds' (date) and 'y' (value).
        
        Returns:
        None
        """
        self.history = data.rename(columns={'date': 'ds', 'value': 'y'})

    def fit(self):
        """
        Fit the Prophet model to the prepared data.
        
        Returns:
        None
        """
        if self.history is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")

        self.model = Prophet()
        self.model.fit(self.history)

    def forecast(self, periods, freq='D'):
        """
        Forecast future values using the fitted Prophet model.
        
        Parameters:
        periods (int): The number of periods to forecast.
        freq (str): Frequency of the forecast ('D' for daily, 'M' for monthly, etc.).
        
        Returns:
        forecast (DataFrame): The forecasted values along with their uncertainty intervals.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def plot_forecast(self, forecast):
        """
        Plot the forecasted values with historical data.
        
        Parameters:
        forecast (DataFrame): The forecasted values returned by the forecast() method.
        
        Returns:
        None
        """
        self.model.plot(forecast)
        plt.title("Forecasted Values")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()

    def evaluate(self, actual_data, forecast):
        """
        Evaluate the forecast by calculating the Mean Squared Error (MSE).
        
        Parameters:
        actual_data (DataFrame): The actual test dataset with two columns: 'ds' (date) and 'y' (value).
        forecast (DataFrame): The forecasted values returned by the forecast() method.
        
        Returns:
        mse (float): The mean squared error between the actual and forecasted values.
        """
        actual = actual_data.rename(columns={'date': 'ds', 'value': 'y'})
        merged = actual.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
        mse = mean_squared_error(merged['y'], merged['yhat'])
        print(f'Mean Squared Error: {mse}')
        return mse

# Example usage:
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.DataFrame({'date': time, 'value': np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500)})
    train_data = data[:400]
    test_data = data[400:]

    # Create Prophet forecasting object
    model = ProphetForecasting()

    # Prepare the data
    model.prepare_data(train_data)

    # Fit the model
    model.fit()

    # Forecast future values
    forecast = model.forecast(periods=len(test_data))

    # Plot the forecast
    model.plot_forecast(forecast)

    # Evaluate the forecast
    model.evaluate(test_data, forecast)
