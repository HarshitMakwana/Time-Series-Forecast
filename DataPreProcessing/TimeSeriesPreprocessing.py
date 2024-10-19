import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

class TimeSeriesPreprocessing:
    def __init__(self, data, target_column, scale_type='minmax'):
        """
        Initialize the Time Series Preprocessing class.

        Parameters:
        data (DataFrame): Input time series data.
        target_column (str): Column name of the target variable.
        scale_type (str): Type of scaling to apply ('minmax' or 'standard').
        """
        self.data = data
        self.target_column = target_column
        self.scale_type = scale_type
        self.scaler = None
        self.features = None
        self.target = None

        # Ensure that the index is a datetime type
        self._ensure_datetime_index()

    def _ensure_datetime_index(self):
        """Ensure the DataFrame index is a datetime type."""
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            raise ValueError("DataFrame index must be a datetime type.")

    def clean_data(self):
        """
        Clean the time series data by removing duplicates and filling missing values.

        Returns:
        None
        """
        # Remove duplicates
        self.data = self.data.drop_duplicates()

        # Fill missing values with interpolation
        self.data = self.data.interpolate()

    def scale_data(self):
        """
        Scale the features and target variable using the specified scaling method.

        Returns:
        None
        """
        if self.scale_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scale_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("scale_type must be either 'minmax' or 'standard'.")

        # Fit the scaler only on the features, excluding the target column
        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]

        self.features_scaled = self.scaler.fit_transform(self.features)
        self.target_scaled = self.scaler.fit_transform(self.target.values.reshape(-1, 1))

    def create_time_windows(self, window_size):
        """
        Create time windows for supervised learning.

        Parameters:
        window_size (int): Size of the time window.

        Returns:
        X (array): Features for model training.
        y (array): Target variable for model training.
        """
        X, y = [], []
        for i in range(len(self.features_scaled) - window_size):
            X.append(self.features_scaled[i:i + window_size])
            y.append(self.target_scaled[i + window_size])

        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2):
        """
        Split the dataset into training and testing sets.

        Parameters:
        X (array): Features for model training.
        y (array): Target variable for model training.
        test_size (float): Proportion of the dataset to include in the test split.

        Returns:
        X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def visualize_data(self):
        """
        Visualize the original and scaled time series data.

        Returns:
        None
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target_column], label='Original Data', color='blue')
        plt.title('Original Time Series Data')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.target_scaled, label='Scaled Data', color='red')
        plt.title('Scaled Time Series Data')
        plt.xlabel('Date')
        plt.ylabel('Scaled ' + self.target_column)
        plt.legend()
        plt.show()

    def seasonal_decomposition(self, model='additive'):
        """
        Decompose the time series data into trend, seasonal, and residual components.

        Parameters:
        model (str): Type of seasonal decomposition ('additive' or 'multiplicative').

        Returns:
        None
        """
        decomposition = seasonal_decompose(self.data[self.target_column], model=model)
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

    def handle_outliers(self):
        """
        Handle outliers in the target variable using Z-score method.

        Returns:
        None
        """
        z_scores = np.abs(stats.zscore(self.data[self.target_column]))
        self.data = self.data[(z_scores < 3)]  # Retain only those rows where Z-score < 3

    def create_lag_features(self, lags):
        """
        Create lag features for the target variable.

        Parameters:
        lags (list): List of lag periods to create features for.

        Returns:
        None
        """
        for lag in lags:
            self.data[f'lag_{lag}'] = self.data[self.target_column].shift(lag)

        # Drop rows with NaN values created by lagging
        self.data = self.data.dropna()

    def encode_categorical_variables(self):
        """
        Encode categorical variables if they exist in the dataset.

        Returns:
        None
        """
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.data[column] = self.data[column].astype('category').cat.codes

    def add_rolling_statistics(self, window_size):
        """
        Add rolling statistics (mean and std) as new features.

        Parameters:
        window_size (int): The window size for rolling calculations.

        Returns:
        None
        """
        self.data[f'rolling_mean_{window_size}'] = self.data[self.target_column].rolling(window=window_size).mean()
        self.data[f'rolling_std_{window_size}'] = self.data[self.target_column].rolling(window=window_size).std()

    def create_time_based_features(self):
        """
        Create time-based features from the datetime index.

        Returns:
        None
        """
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['hour'] = self.data.index.hour

    def difference(self):
        """
        Difference the time series data to remove trends.

        Returns:
        None
        """
        self.data[self.target_column] = self.data[self.target_column].diff()
        self.data = self.data.dropna()

    def slice_data(self, start_date, end_date):
        """
        Slice the data between specified start and end dates.

        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
        DataFrame: Sliced DataFrame.
        """
        return self.data.loc[start_date:end_date]

    def feature_selection(self, threshold):
        """
        Select features based on correlation with the target variable.

        Parameters:
        threshold (float): Correlation threshold for feature selection.

        Returns:
        None
        """
        corr = self.data.corr()
        self.features = corr[abs(corr[self.target_column]) > threshold].index.tolist()
        self.data = self.data[self.features + [self.target_column]]

# Example usage
if __name__ == "__main__":
    # Create example time series data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # Add categorical feature for demonstration
    categorical_feature = np.random.choice(['A', 'B', 'C'], size=500)

    data = pd.DataFrame({
        'Date': dates,
        'Value': np.sin(2 * np.pi * dates.dayofyear / 50) + np.random.normal(0, 0.1, 500),
        'Category': categorical_feature
    }).set_index('Date')

    ts_preprocessing = TimeSeriesPreprocessing(data, target_column='Value', scale_type='minmax')
    ts_preprocessing.clean_data()
    ts_preprocessing.encode_categorical_variables()
    ts_preprocessing.add_rolling_statistics(window_size=7)
    ts_preprocessing.create_time_based_features()
    ts_preprocessing.difference()
    
    # Split and prepare data
    ts_preprocessing.scale_data()
    X, y = ts_preprocessing.create_time_windows(window_size=5)
    X_train, X_test, y_train, y_test = ts_preprocessing.split_data(X, y)

    # Visualize the data
    ts_preprocessing.visualize_data()

    # Seasonal decomposition
    ts_preprocessing.seasonal_decomposition()
