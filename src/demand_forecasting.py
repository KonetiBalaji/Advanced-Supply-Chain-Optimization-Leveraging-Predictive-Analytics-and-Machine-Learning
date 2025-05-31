# src/demand_forecasting.py

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecaster:
    def __init__(self, data: pd.DataFrame, date_column: str, target_column: str):
        """
        Initialize the DemandForecaster.
        
        Args:
            data: DataFrame containing the time series data
            date_column: Name of the date column
            target_column: Name of the target column to forecast
        """
        self.data = data.copy()
        self.date_column = date_column
        self.target_column = target_column
        self.model = None
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare the data for forecasting."""
        try:
            # Convert date column to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            
            # Set date as index
            self.data.set_index(self.date_column, inplace=True)
            
            # Sort by date
            self.data.sort_index(inplace=True)
            
            # Resample to monthly frequency and sum
            self.monthly_data = self.data[self.target_column].resample('ME').sum()
            
            # Remove any outliers (values more than 3 standard deviations from mean)
            mean = self.monthly_data.mean()
            std = self.monthly_data.std()
            self.monthly_data = self.monthly_data[
                (self.monthly_data >= mean - 3*std) & 
                (self.monthly_data <= mean + 3*std)
            ]
            
            # Log transform to handle large values
            self.monthly_data = np.log1p(self.monthly_data)
            
            # Scale the data to a more manageable range (0-100)
            self.scale_factor = 100 / self.monthly_data.max()
            self.monthly_data = self.monthly_data * self.scale_factor
            
            logger.info(f"Data prepared successfully for forecasting. Shape: {self.monthly_data.shape}")
            logger.info(f"Date range: {self.monthly_data.index.min()} to {self.monthly_data.index.max()}")
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def fit(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Fit the SARIMA model to the data.
        
        Args:
            order: The (p,d,q) order of the model
            seasonal_order: The (P,D,Q,s) seasonal order of the model
        """
        try:
            logger.info(f"Fitting SARIMA model with order {order} and seasonal order {seasonal_order}")
            
            # Ensure we have enough data for seasonal components
            if len(self.monthly_data) < 24:  # Need at least 2 years of data
                logger.warning("Insufficient data for seasonal components. Using simpler model.")
                seasonal_order = (0, 0, 0, 12)
            
            # Fit SARIMA model
            self.model = SARIMAX(
                self.monthly_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.model_fit = self.model.fit(disp=False)
            logger.info("SARIMA model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            raise
    
    def forecast_demand(self, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate demand forecast.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast values, forecast intervals)
        """
        try:
            if self.model_fit is None:
                raise ValueError("Model must be fitted before forecasting")
            
            # Generate forecast
            forecast = self.model_fit.forecast(steps=steps)
            
            # Get forecast intervals
            forecast_intervals = self.model_fit.get_forecast(steps=steps).conf_int()
            
            # Rescale the forecast back to original scale
            forecast = forecast / self.scale_factor
            forecast_intervals = forecast_intervals / self.scale_factor
            
            # Reverse log transform
            forecast = np.expm1(forecast)
            forecast_intervals = np.expm1(forecast_intervals)
            
            logger.info("Generated forecast successfully")
            return forecast, forecast_intervals
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def calculate_seasonal_factors(self) -> Dict[str, float]:
        """
        Calculate seasonal factors for each month.
        
        Returns:
            Dictionary of monthly seasonal factors
        """
        try:
            # Get the seasonal decomposition
            decomposition = seasonal_decompose(self.monthly_data, period=12)
            
            # Calculate seasonal factors
            seasonal_factors = decomposition.seasonal[:12]
            
            # Normalize factors to be around 1.0
            seasonal_factors = seasonal_factors / seasonal_factors.mean()
            
            # Convert to dictionary with month names
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            seasonal_dict = {
                month: float(factor)
                for month, factor in zip(month_names, seasonal_factors)
            }
            
            logger.info("Seasonal factors calculated successfully")
            return seasonal_dict
            
        except Exception as e:
            logger.error(f"Error calculating seasonal factors: {str(e)}")
            raise
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns:
            String containing model summary
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.summary().as_text()
    
    def plot_forecast(self, forecast: np.ndarray, intervals: Dict[str, np.ndarray] = None):
        """
        Plot the forecast results.
        
        Args:
            forecast: Forecast values
            intervals: Forecast intervals
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.monthly_data.index, self.monthly_data, label='Historical')
            
            # Create forecast dates
            last_date = self.monthly_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=len(forecast),
                freq='M'
            )
            
            plt.plot(forecast_dates, forecast, label='Forecast', color='red')
            
            if intervals is not None:
                plt.fill_between(
                    forecast_dates,
                    intervals['lower'],
                    intervals['upper'],
                    color='red',
                    alpha=0.1
                )
            
            plt.title('Demand Forecast')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            raise
    
    def evaluate_model(self, test_size: int = 12) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            test_size: Number of periods to use for testing
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data
        ts_data = self.monthly_data
        
        # Split data into train and test
        train = ts_data[:-test_size]
        test = ts_data[-test_size:]
        
        # Generate predictions for test period
        predictions = self.model.predict(start=len(train), end=len(train) + test_size - 1)
        
        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
