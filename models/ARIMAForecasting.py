import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ARIMAForecasting:
    def __init__(self, p, d, q):
        """
        Initialize the ARIMA model parameters.
        
        Parameters:
        p (int): The number of lag observations (AR term).
        d (int): The number of times the raw observations are differenced (I term).
        q (int): The size of the moving average window (MA term).
        """
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def train(self, train_data):
        """
        Train the ARIMA model on the provided training data.
        
        Parameters:
        train_data (array-like): The training dataset.
        
        Returns:
        None
        """
        self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())

    def forecast(self, steps=1):
        """
        Forecast future values based on the trained ARIMA model.
        
        Parameters:
        steps (int): Number of future time steps to predict.
        
        Returns:
        forecast (array-like): The forecasted values.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been trained yet!")
        
        forecast = self.model_fit.forecast(steps=steps)
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
    # Simulated example data
    data = np.sin(np.linspace(0, 100, 500)) + np.random.normal(0, 0.5, 500)
    train_data, test_data = data[:400], data[400:]

    # Create ARIMA forecasting object
    model = ARIMAForecasting(p=5, d=1, q=0)

    # Train the model
    model.train(train_data)

    # Forecast future values
    forecast_values = model.forecast(steps=len(test_data))

    # Plot the forecast vs actual values
    model.plot_forecast(test_data, forecast_values)

    # Evaluate the forecast
    model.evaluate(test_data, forecast_values)
