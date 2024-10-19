import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class SARIMAForecasting:
    def __init__(self, p, d, q, P, D, Q, s):
        """
        Initialize the SARIMA model parameters.
        
        Parameters:
        p (int): The number of lag observations for the AR term.
        d (int): The number of times the data is differenced.
        q (int): The order of the MA term.
        P (int): The number of seasonal autoregressive terms.
        D (int): The number of seasonal differencing.
        Q (int): The order of the seasonal moving average.
        s (int): The number of time steps in a seasonal period (seasonal length).
        """
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.model_fit = None

    def train(self, train_data):
        """
        Train the SARIMA model on the provided training data.
        
        Parameters:
        train_data (array-like): The training dataset.
        
        Returns:
        None
        """
        self.model = SARIMAX(train_data, 
                             order=(self.p, self.d, self.q), 
                             seasonal_order=(self.P, self.D, self.Q, self.s))
        self.model_fit = self.model.fit(disp=False)
        print(self.model_fit.summary())

    def forecast(self, steps=1):
        """
        Forecast future values using the trained SARIMA model.
        
        Parameters:
        steps (int): Number of future time steps to predict.
        
        Returns:
        forecast (array-like): The forecasted values.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been trained yet!")

        return self.model_fit.forecast(steps=steps)

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

    # Create SARIMA forecasting object
    model = SARIMAForecasting(p=1, d=1, q=1, P=1, D=1, Q=1, s=50)  # s=50 for seasonality of 50 time steps

    # Train the model
    model.train(train_data)

    # Forecast future values
    forecast_values = model.forecast(steps=len(test_data))

    # Plot the forecast vs actual values
    model.plot_forecast(test_data, forecast_values)

    # Evaluate the forecast
    model.evaluate(test_data, forecast_values)
