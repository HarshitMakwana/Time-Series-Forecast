import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ThetaModelForecasting:
    def __init__(self):
        """
        Initialize the Theta Model Forecasting class.
        """
        self.trend_model = None
        self.seasonal_component = None
        self.mean_value = None
        self.detrended_series = None

    def decompose(self, series, period):
        """
        Decompose the time series into trend, seasonal, and residual components.
        
        Parameters:
        series (array-like): The time series data.
        period (int): The period of seasonality (e.g., 12 for monthly data with yearly seasonality).
        
        Returns:
        decomposition (object): Decomposed time series with trend, seasonal, and residual components.
        """
        self.mean_value = np.mean(series)
        decomposition = seasonal_decompose(series, period=period, model='additive')
        self.seasonal_component = decomposition.seasonal
        self.detrended_series = series - self.seasonal_component
        return decomposition

    def fit_trend(self, series):
        """
        Fit a linear regression model to the trend component of the series.
        
        Parameters:
        series (array-like): The detrended time series data.
        
        Returns:
        None
        """
        # Create time index
        time = np.arange(len(series)).reshape(-1, 1)
        self.trend_model = LinearRegression()
        self.trend_model.fit(time, series)

    def forecast(self, steps=1):
        """
        Forecast future values using the Theta model.
        
        Parameters:
        steps (int): Number of future time steps to predict.
        
        Returns:
        forecast (array-like): The forecasted values.
        """
        if self.trend_model is None or self.seasonal_component is None:
            raise ValueError("Model is not fully initialized. Ensure you have decomposed the series and fit the trend.")

        # Time index for future periods
        time = np.arange(len(self.detrended_series), len(self.detrended_series) + steps).reshape(-1, 1)
        
        # Predict trend
        trend_forecast = self.trend_model.predict(time)
        
        # Cycle through seasonal component (repeat for the number of forecast steps)
        seasonal_forecast = np.tile(self.seasonal_component[:steps], int(np.ceil(steps / len(self.seasonal_component))))[:steps]
        
        # Combine trend and seasonality, and add the mean
        forecast = trend_forecast + seasonal_forecast + self.mean_value
        
        return forecast

    def plot_forecast(self, test_data, forecast_values):
        """
        Plot the actual vs forecasted values for visualization.
        
        Parameters:
        test_data (array-like): The actual test dataset.
        forecast_values (array-like): The forecasted values.
        
        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(test_data, label='Actual')
        plt.plot(forecast_values, label='Forecast', color='red')
        plt.title('Actual vs Forecasted values')
        plt.legend()
        plt.show()

    def evaluate(self, test_data, forecast_values):
        """
        Evaluate the forecast by calculating the Mean Squared Error (MSE).
        
        Parameters:
        test_data (array-like): The actual test dataset.
        forecast_values (array-like): The forecasted values.
        
        Returns:
        mse (float): The mean squared error between the actual and forecasted values.
        """
        mse = mean_squared_error(test_data, forecast_values)
        print(f'Mean Squared Error: {mse}')
        return mse

# Example usage:
if __name__ == "__main__":
    # Simulated example data (seasonal sine wave data with noise)
    time = np.arange(500)
    data = np.sin(2 * np.pi * time / 50) + np.random.normal(0, 0.5, 500)  # Seasonality with period 50
    train_data, test_data = data[:400], data[400:]

    # Create Theta model forecasting object
    model = ThetaModelForecasting()

    # Decompose the data (assuming a seasonality period of 50)
    model.decompose(train_data, period=50)

    # Fit the trend component
    model.fit_trend(model.detrended_series)

    # Forecast future values
    forecast_values = model.forecast(steps=len(test_data))

    # Plot the forecast vs actual values
    model.plot_forecast(test_data, forecast_values)

    # Evaluate the forecast
    model.evaluate(test_data, forecast_values)
