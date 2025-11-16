"""
Data Processing Module for Hanoi Weather Data
Handles data cleaning, preprocessing, and preparation for feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import os
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings('ignore')


class WeatherDataProcessor:
    """
    Main class for processing Hanoi weather data.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.outlier_summary = {}
    
    def generate_output_path(self, input_path: str, suffix: str) -> str:
        """
        Generate output path based on input path.
        
        Args:
            input_path: Path to input file
            suffix: Suffix to add (e.g., '_clean', '_engineered')
            
        Returns:
            Output path with suffix added before file extension
        """
        # Split path and filename
        dir_path = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # Create output filename
        output_filename = f"{name}{suffix}{ext}"
        
        # Replace 'raw' with 'processed' in directory path if exists
        if 'raw' in dir_path:
            output_dir = dir_path.replace('raw', 'processed')
        else:
            output_dir = dir_path
            
        output_path = os.path.join(output_dir, output_filename)
        
        return output_path
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw weather data from CSV file."""
        if self.verbose:
            print(f"Loading data from: {file_path}")
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        if self.verbose:
            print(f"Data loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        if self.verbose:
            print("\nHANDLING MISSING VALUES:")
        
        # Handle preciptype
        if 'preciptype' in df.columns:
            missing_preciptype = df['preciptype'].isnull().sum()
            if missing_preciptype > 0:
                df['preciptype'] = df['preciptype'].fillna('none')
                if self.verbose:
                    print(f"Preciptype: {missing_preciptype} NaN values filled with 'none'")
        
        # Handle severerisk
        if 'severerisk' in df.columns:
            missing_severerisk = df['severerisk'].isnull().sum()
            if missing_severerisk > 0:
                df['severerisk'] = df['severerisk'].fillna(0)
                if self.verbose:
                    print(f"Severerisk: {missing_severerisk} NaN values filled with 0")
        
        # Clean invalid values
        if 'precip' in df.columns:
            df['precip'] = df['precip'].clip(lower=0)
            
        if 'humidity' in df.columns:
            df['humidity'] = df['humidity'].clip(upper=100)
            
        if 'cloudcover' in df.columns:
            df['cloudcover'] = df['cloudcover'].clip(upper=100)
        
        # Check remaining missing values
        remaining_missing = df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        
        if len(remaining_missing) > 0 and self.verbose:
            print(f"Remaining missing values:")
            for col, count in remaining_missing.items():
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
        elif self.verbose:
            print("All missing values handled!")
        
        return df
    
    def detect_outliers_iqr(self, data: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple:
        """Detect outliers using IQR method."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def analyze_outliers(self, df: pd.DataFrame) -> Dict:
        """Analyze outliers in numerical columns."""
        numerical_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'precip', 
                         'windspeed', 'sealevelpressure', 'cloudcover', 'visibility']
        
        outlier_summary = {}
        
        if self.verbose:
            print("\nOUTLIER ANALYSIS:")
        
        for col in numerical_cols:
            if col in df.columns:
                outliers, lower, upper = self.detect_outliers_iqr(df, col)
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(df)) * 100
                
                outlier_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
                
                if self.verbose:
                    print(f"{col:15}: {outlier_count:4d} outliers ({outlier_pct:5.1f}%)")
        
        self.outlier_summary = outlier_summary
        return outlier_summary
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic temporal features."""
        df = df.copy()
        
        if self.verbose:
            print("\nCREATING TEMPORAL FEATURES:")
        
        # Basic time components
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        
        # Season mapping
        season_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        
        df['season'] = df['month'].map(season_mapping)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rainy'] = (df['precip'] > 0).astype(int)
        
        # Temperature range
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        temporal_features = ['year', 'month', 'day', 'day_of_year', 'day_of_week', 
                           'week_of_year', 'season', 'is_weekend', 'is_rainy', 'temp_range']
        
        if self.verbose:
            created_features = [f for f in temporal_features if f in df.columns]
            print(f"Created {len(created_features)} temporal features")
            for feature in created_features:
                print(f"  - {feature}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for modeling."""
        df = df.copy()
        
        if self.verbose:
            print("\nCREATING ADVANCED FEATURES:")
        
        # Sort by datetime to ensure proper lag calculation
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Lag features for time series
        lag_days = [1, 2, 3, 7]
        for lag in lag_days:
            df[f'temp_lag{lag}'] = df['temp'].shift(lag)
            
        if 'tempmax' in df.columns:
            df['tempmax_lag1'] = df['tempmax'].shift(1)
            
        if 'tempmin' in df.columns:
            df['tempmin_lag1'] = df['tempmin'].shift(1)
        
        # Rolling statistics
        windows = [3, 7, 14]
        for window in windows:
            df[f'temp_rolling_mean_{window}d'] = df['temp'].rolling(window=window, min_periods=1).mean()
            df[f'temp_rolling_std_{window}d'] = df['temp'].rolling(window=window, min_periods=1).std()
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Weather-derived features
        if 'winddir' in df.columns:
            df['wind_dir_sin'] = np.sin(np.deg2rad(df['winddir']))
            df['wind_dir_cos'] = np.cos(np.deg2rad(df['winddir']))
        
        # Comfort indices
        if all(col in df.columns for col in ['tempmax', 'tempmin', 'humidity']):
            # Simple discomfort index
            df['discomfort_index'] = 0.72 * (df['tempmax'] + df['tempmin']) + 0.4 * df['humidity']
        
        if self.verbose:
            new_features = [col for col in df.columns if any(x in col for x in ['lag', 'rolling', 'sin', 'cos', '_dir_', 'discomfort'])]
            print(f"Created {len(new_features)} advanced features")
        
        return df
    
    def visualize_outliers(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize outliers for key variables."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Outlier Detection - Key Weather Variables', fontsize=16)
        
        variables_to_plot = ['temp', 'humidity', 'precip', 'windspeed']
        for i, var in enumerate(variables_to_plot):
            if var in df.columns and var in self.outlier_summary:
                row, col = i // 2, i % 2
                
                # Box plot
                df.boxplot(column=var, ax=axes[row, col])
                axes[row, col].set_title(f'{var.title()} - {self.outlier_summary[var]["count"]} outliers')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Outlier visualization saved to: {save_path}")
        
        plt.show()
    
    def export_data(self, df: pd.DataFrame, output_path: str):
        """Export processed data to CSV."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\nData exported to: {output_path}")
            print(f"Final dataset shape: {df.shape}")
    
    def process_basic_clean(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Basic cleaning only (no advanced features).
        If output_path is None, auto-generate based on input_path.
        """
        # Auto-generate output path if not provided
        if output_path is None:
            output_path = self.generate_output_path(input_path, '_cleaned')
        
        if self.verbose:
            print("BASIC DATA CLEANING PIPELINE")
        
        # Step 1: Load data
        df = self.load_data(input_path)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Analyze outliers
        self.analyze_outliers(df)
        
        # Step 4: Basic temporal features only
        df = self.create_temporal_features(df)
        
        # Step 5: Export processed data
        self.export_data(df, output_path)
        
        if self.verbose:
            print(f"\nBASIC CLEANING COMPLETED!")
        
        return df
    
    def process_with_features(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete processing with feature engineering.
        If output_path is None, auto-generate based on input_path.
        """
        # Auto-generate output path if not provided
        if output_path is None:
            output_path = self.generate_output_path(input_path, '_engineered')
        
        if self.verbose:
            print("COMPLETE DATA PROCESSING WITH FEATURE ENGINEERING")
        
        # Step 1: Load data
        df = self.load_data(input_path)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Analyze outliers
        self.analyze_outliers(df)
        
        # Step 4: Basic temporal features
        df = self.create_temporal_features(df)
        
        # Step 5: Advanced features
        df = self.create_advanced_features(df)
        
        # Step 6: Drop rows with NaN from lag features
        if self.verbose:
            before_dropna = len(df)
        df = df.dropna().reset_index(drop=True)
        if self.verbose:
            after_dropna = len(df)
            print(f"\nDropped {before_dropna - after_dropna} rows with NaN from lag features")
        
        # Step 7: Export processed data
        self.export_data(df, output_path)
        
        if self.verbose:
            print(f"\nCOMPLETE PROCESSING WITH FEATURES COMPLETED!")
            print(f"Ready for Modeling phase")
        
        return df
    
    def process_data(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            input_path: Path to raw data CSV file
            output_path: Path to save processed data (optional)
            
        Returns:
            Processed DataFrame
        """
        return self.process_basic_clean(input_path, output_path)


# Updated utility functions
def quick_clean(input_path: str, output_path: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Quick basic cleaning function with auto output path."""
    processor = WeatherDataProcessor(verbose=verbose)
    return processor.process_basic_clean(input_path, output_path)

def quick_process_full(input_path: str, output_path: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Quick full processing with feature engineering and auto output path."""
    processor = WeatherDataProcessor(verbose=verbose)
    return processor.process_with_features(input_path, output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        # Command line usage
        input_file = sys.argv[1]
        process_type = sys.argv[2] if len(sys.argv) > 2 else "clean"
        
        processor = WeatherDataProcessor(verbose=True)
        
        if process_type.lower() == "full":
            df = processor.process_with_features(input_file)
        else:
            df = processor.process_basic_clean(input_file)
            
        print(f"\nProcessing completed!")
        
    else:
        # Default behavior - process multiple files
        input_files = [
            "../../data/raw/daily_data.csv",
            "../../data/raw/hourly_data.csv"  # Example
        ]
        
        processor = WeatherDataProcessor(verbose=True)
        
        for input_file in input_files:
            if os.path.exists(input_file):
                print(f"\nProcessing: {input_file}")
                
                # Basic cleaning (auto output: daily_data_clean.csv)
                df_clean = processor.process_basic_clean(input_file)
                
                # Full processing (auto output: daily_data_engineered.csv)
                df_features = processor.process_with_features(input_file)
            else:
                print(f"File not found: {input_file}")