import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FourierForecast:
    def __init__(self, n_components):
        """
        Initialize the Fourier Transform forecasting class.

        Parameters:
        n_components (int): Number of Fourier components to use for forecasting.
        """
        self.n_components = n_components
        self.coefficients = None

    def fit(self, data):
        """
        Fit the Fourier model to the data.

        Parameters:
        data (array-like): The time series data.

        Returns:
        None
        """
        self.coefficients = np.fft.fft(data)

    def forecast(self, steps):
        """
        Forecast future values using the Fourier components.

        Parameters:
        steps (int): Number of steps to forecast.

        Returns:
        array: Forecasted values.
        """
        freq = np.fft.fftfreq(len(self.coefficients))
        forecast = np.zeros(steps)
        for i in range(self.n_components):
            forecast += self.coefficients[i] * np.cos(2 * np.pi * freq[i] * np.arange(steps))
        return forecast.real

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
        plt.title("Fourier Transform Time Series Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data (time series data)
    time = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.Series(np.sin(2 * np.pi * time.dayofyear / 50) + np.random.normal(0, 0.1, 500), index=time)

    # Fit Fourier model and forecast
    fourier_model = FourierForecast(n_components=10)
    fourier_model.fit(data.values)
    forecast = fourier_model.forecast(30)

    # Plot results
    fourier_model.plot(data, forecast)
