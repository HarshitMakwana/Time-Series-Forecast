import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

class WaveletForecast:
    def __init__(self, wavelet='haar'):
        """
        Initialize the Wavelet Transform forecasting class.

        Parameters:
        wavelet (str): Name of the wavelet to use for decomposition.
        """
        self.wavelet = wavelet
        self.coefficients = None

    def fit(self, data):
        """
        Fit the Wavelet model to the data.

        Parameters:
        data (array-like): The time series data.

        Returns:
        None
        """
        self.coefficients = pywt.wavedec(data, self.wavelet)

    def forecast(self, steps):
        """
        Forecast future values using the wavelet coefficients.

        Parameters:
        steps (int): Number of steps to forecast.

        Returns:
        array: Forecasted values.
        """
        # Using the last approximation coefficients for forecasting
        approximation = self.coefficients[0]
        return np.tile(approximation[-1], steps)

    def plot(self, data, forecast):
        """
        Plot the actual data and forecasted values.

        Parameters:
        data (array-like): Actual time series data.
        forecast (array): Forecasted values.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data, label='Actual', color='blue')
        future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(forecast))
        plt.plot(future_index, forecast, label='Forecasted', color='red')
        plt.title("Wavelet Transform Time Series Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)

    # Fit Wavelet model and forecast
    wavelet_model = WaveletForecast(wavelet='haar')
    wavelet_model.fit(data.values)
    forecast = wavelet_model.forecast(30)

    # Plot results
    wavelet_model.plot(data, forecast)
