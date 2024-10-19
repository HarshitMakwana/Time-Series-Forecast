import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan

class TimeSeriesBSTS:
    def __init__(self, seasonal_periods=None, n_changepoints=5):
        """
        Initialize the BSTS model for time series forecasting.

        Parameters:
        seasonal_periods (int): Number of periods in each season (e.g., 12 for monthly data).
        n_changepoints (int): Number of changepoints in the trend component.
        """
        self.seasonal_periods = seasonal_periods
        self.n_changepoints = n_changepoints
        self.model = None
        self.fit = None

    def _stan_model(self):
        """
        Define the Stan model for BSTS.
        """
        model_code = """
        data {
            int<lower=0> N;
            int<lower=0> seasonal_periods;
            vector[N] y;
        }
        parameters {
            vector[N] level;
            vector[N] seasonal;
            vector[N] trend;
            real<lower=0> sigma;
        }
        model {
            for (n in 2:N) {
                level[n] ~ normal(level[n-1], sigma);
                trend[n] ~ normal(trend[n-1], sigma);
                seasonal[n] ~ normal(seasonal[n - seasonal_periods], sigma);
                y[n] ~ normal(level[n] + trend[n] + seasonal[n], sigma);
            }
        }
        """
        return pystan.StanModel(model_code=model_code)

    def train(self, data):
        """
        Train the BSTS model on the provided data.

        Parameters:
        data (array-like): The time series data.

        Returns:
        None
        """
        self.model = self._stan_model()
        self.fit = self.model.sampling(data={'N': len(data), 
                                              'seasonal_periods': self.seasonal_periods,
                                              'y': data},
                                       iter=1000, chains=4)

    def predict(self, steps):
        """
        Predict future values using the trained model.

        Parameters:
        steps (int): Number of future steps to predict.

        Returns:
        predictions (array): Predicted values.
        """
        if self.fit is None:
            raise ValueError("Model has not been trained yet.")

        predictions = self.fit.extract()['level'][-1] + self.fit.extract()['trend'][-1]
        return predictions[-steps:]

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
        plt.title("BSTS Time Series Forecasting")
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
    model = TimeSeriesBSTS(seasonal_periods=50)
    model.train(data)

    # Predict future values
    steps = 10
    predictions = model.predict(steps)

    # Plot the results
    model.plot(data, predictions)
