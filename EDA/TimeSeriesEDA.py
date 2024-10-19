import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm

class TimeSeriesEDA:
    def __init__(self, data, target_column):
        """
        Initialize the Time Series EDA class.

        Parameters:
        data (DataFrame): Input time series data.
        target_column (str): Column name of the target variable.
        """
        self.data = data
        self.target_column = target_column

    def visualize_time_series(self):
        """
        Visualize the time series data.

        Returns:
        None
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target_column], color='blue')
        plt.title('Time Series Data')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.grid()
        plt.show()

    def summary_statistics(self):
        """
        Print summary statistics of the time series data.

        Returns:
        None
        """
        print("Summary Statistics:")
        print(self.data.describe())

    def check_missing_values(self):
        """
        Check for missing values in the dataset.

        Returns:
        None
        """
        missing_values = self.data.isnull().sum()
        print("Missing Values:")
        print(missing_values[missing_values > 0])

    def adf_test(self):
        """
        Perform Augmented Dickey-Fuller test to check for stationarity.

        Returns:
        None
        """
        result = adfuller(self.data[self.target_column])
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        for key, value in result[4].items():
            print(f'Critical Value {key}: {value}')
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")

    def plot_acf_pacf(self, lags=40):
        """
        Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

        Parameters:
        lags (int): Number of lags to include in the plots.

        Returns:
        None
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(self.data[self.target_column], lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function')
        plot_pacf(self.data[self.target_column], lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function')
        plt.show()

    def visualize_seasonality(self):
        """
        Visualize seasonal decomposition of time series.

        Returns:
        None
        """
        decomposition = seasonal_decompose(self.data[self.target_column], model='additive', period=12)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(self.data[self.target_column], label='Original')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(residual, label='Residual')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def visualize_correlation(self):
        """
        Visualize the correlation matrix of the dataset.

        Returns:
        None
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
        plt.title('Correlation Matrix')
        plt.show()

    def create_lagged_features(self, lags):
        """
        Create lagged features for the target variable.

        Parameters:
        lags (list): List of lag periods to create features for.

        Returns:
        DataFrame: DataFrame with lagged features added.
        """
        lagged_data = self.data.copy()
        for lag in lags:
            lagged_data[f'lag_{lag}'] = lagged_data[self.target_column].shift(lag)
        return lagged_data.dropna()

    def check_normality(self):
        """
        Check if the data follows a normal distribution.

        Returns:
        None
        """
        stat, p = normaltest(self.data[self.target_column])
        print('Normality Test Statistic:', stat)
        print('p-value:', p)
        if p > 0.05:
            print("The data follows a normal distribution.")
        else:
            print("The data does not follow a normal distribution.")

    def outlier_detection(self):
        """
        Detect outliers in the dataset using Z-score.

        Returns:
        None
        """
        z_scores = np.abs((self.data[self.target_column] - self.data[self.target_column].mean()) / self.data[self.target_column].std())
        outliers = np.where(z_scores > 3)
        print("Detected Outliers at indices:", outliers)

    def scale_data(self, scale_type='minmax'):
        """
        Scale the time series data using Min-Max or Standard scaling.

        Parameters:
        scale_type (str): 'minmax' or 'standard' for scaling.

        Returns:
        None
        """
        if scale_type == 'minmax':
            scaler = MinMaxScaler()
            self.data[self.target_column] = scaler.fit_transform(self.data[[self.target_column]])
        elif scale_type == 'standard':
            scaler = StandardScaler()
            self.data[self.target_column] = scaler.fit_transform(self.data[[self.target_column]])
        else:
            print("Invalid scale type. Use 'minmax' or 'standard'.")

    def create_time_features(self):
        """
        Create additional time-based features.

        Returns:
        None
        """
        self.data['Year'] = self.data.index.year
        self.data['Month'] = self.data.index.month
        self.data['Day'] = self.data.index.day
        self.data['Weekday'] = self.data.index.weekday
        self.data['Quarter'] = self.data.index.quarter
        print("Time-based features added.")

# Example usage
if __name__ == "__main__":
    # Create example time series data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Value': np.sin(2 * np.pi * dates.dayofyear / 50) + np.random.normal(0, 0.1, 500)
    }).set_index('Date')

    ts_eda = TimeSeriesEDA(data, target_column='Value')
    ts_eda.visualize_time_series()
    ts_eda.summary_statistics()
    ts_eda.check_missing_values()
    ts_eda.adf_test()
    ts_eda.plot_acf_pacf()
    ts_eda.visualize_seasonality()
    ts_eda.visualize_correlation()

    lagged_data = ts_eda.create_lagged_features(lags=[1, 2, 3])
    print(lagged_data.head())

    ts_eda.check_normality()
    ts_eda.outlier_detection()
    ts_eda.scale_data(scale_type='minmax')
    ts_eda.create_time_features()
    print(ts_eda.data.head())
