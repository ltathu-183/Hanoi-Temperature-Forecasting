import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn
import os
from pathlib import Path

def handle_sklearn_compatibility():
    """Handle scikit-learn version compatibility issues"""
    try:
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            class _RemainderColsList(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
            print("Patched _RemainderColsList for sklearn compatibility")
    except Exception as e:
        print(f"Compatibility patch failed: {e}")

# Apply compatibility fix when module loads
handle_sklearn_compatibility()

def get_model_paths():
    """Get correct paths for model files"""
    try:
        project_root = Path(__file__).parent.parent
        model_file = project_root / "models" / "daily" / "BEST_CATBOOST_TIMESERIES.joblib"
        selection_file = project_root / "models" / "daily" / "selection_result.joblib"
        
        print(f"Looking for model in: {model_file}")
        print(f"Model exists: {model_file.exists()}")
        print(f"Selection exists: {selection_file.exists()}")
        
        return model_file, selection_file
    except Exception as e:
        print(f"Error getting model paths: {e}")
        return None, None

def process_data(df: pd.DataFrame):
    """Process the weather data for feature engineering - FIXED fragmentation"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    df['datetime'] = pd.to_datetime(df['datetime']) 
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Create season mapping
    season_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

    df['season'] = df['month'].map(season_mapping)

    # Temperature range
    df['temp_range'] = df['tempmax'] - df['tempmin']

    # FIXED: Use df.assign() and avoid inplace operations
    # Remove unused columns - do this in one operation
    columns_to_drop = ['name', 'description', 'icon', 'preciptype', 'snow', 
                      'snowdepth', 'stations', 'severerisk', 'conditions']
    df = df.drop(columns=columns_to_drop)

    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['day_length_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600.0
    df = df.drop(columns=['sunrise', 'sunset'])

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Drop multiple columns at once
    df = df.drop(columns=['day', 'month', 'day_of_year'])

    # Interaction features - build a dictionary first, then assign all at once
    interaction_features = {
        "temp_solar_interaction": df["temp"] * df["solarradiation"],
        "uv_temp_interaction": df["uvindex"] * df["temp"],
        'temp_cloudcover_interaction': df['temp'] * df['cloudcover'],
        'temp_sealevelpressure_interaction': df['temp'] * df['sealevelpressure'],
        'weighted_precip': df['precipprob'] * df['precip'],
        'effective_solar': df['solarradiation'] * (1 - df['cloudcover']/100),
        'precip_impact': df['precipprob'] * df['precip']
    }
    
    # Add all interaction features at once
    df = df.assign(**interaction_features)

    # Wind features
    df['wind_u'] = df['windspeed'] * np.cos(2 * np.pi * df['winddir'] / 360)
    df['wind_v'] = df['windspeed'] * np.sin(2 * np.pi * df['winddir'] / 360)
    df = df.drop('winddir', axis=1)

    temp_minus_dew = df['temp'] - df['dew']

    # Moonphase features
    df['moonphase_sin'] = np.sin(2 * np.pi * df['moonphase'] / 1)
    df['moonphase_cos'] = np.cos(2 * np.pi * df['moonphase'] / 1)
    df = df.drop('moonphase', axis=1)

    # FIXED: Create lag features without fragmentation
    def create_lag_features(df, cols, lags):
        """Create lag features without DataFrame fragmentation"""
        lag_data = {}
        for col in cols:
            for lag in lags:
                lag_data[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Add all lag features at once
        return df.assign(**lag_data)

    # Specify columns and lags
    computing_columns = df.drop(columns=['year', 'season', 'month_sin',
                                        'month_cos', 'dayofyear_sin', 'dayofyear_cos']).columns

    lag_steps = [1, 2, 3, 5, 7, 10, 14, 21, 30]

    # Apply lagging features
    df = create_lag_features(df, computing_columns, lag_steps)

    # FIXED: Rolling features without fragmentation
    def compute_rolling_all(df, horizons, cols):
        """Compute all rolling features at once to avoid fragmentation"""
        rolling_data = {}
        
        for horizon in horizons:
            for col in cols:
                rolling_mean = df[col].rolling(horizon, min_periods=horizon).mean()
                label = f"rolling_{horizon}_{col}"
                rolling_data[label] = rolling_mean
                rolling_data[f"{label}_change"] = df[col] - rolling_mean
        
        return df.assign(**rolling_data)

    # Compute rolling features for specified horizons
    rolling_horizons = [3, 7, 14, 21, 30]
    df = compute_rolling_all(df, rolling_horizons, computing_columns)
    
    # FIXED: Expanding mean features without fragmentation
    def add_expanding_features(df, computing_columns):
        """Add all expanding mean features at once"""
        expanding_data = {}
        
        for col in computing_columns:
            expanding_data[f"month_avg_{col}"] = df[col].groupby(df.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())
            expanding_data[f"day_avg_{col}"] = df[col].groupby(df.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())
            expanding_data[f"year_avg_{col}"] = df[col].groupby(df.index.year, group_keys=False).apply(lambda x: x.expanding(1).mean())
            expanding_data[f"season_avg_{col}"] = df[col].groupby(df['season'], group_keys=False).apply(lambda x: x.expanding(1).mean())
        
        # Add temperature min/max features
        expanding_data["month_max_temp"] = df['temp'].groupby(df.index.month, group_keys=False).cummax()
        expanding_data["month_min_temp"] = df['temp'].groupby(df.index.month, group_keys=False).cummin()
        
        return df.assign(**expanding_data)

    df = add_expanding_features(df, computing_columns)

    # FIXED: Volatility and anomaly features without fragmentation
    volatility_data = {
        "temp_volatility_7": df["temp"].rolling(7).std(),
        "temp_volatility_14": df["temp"].rolling(14).std(),
        "temp_volatility_21": df["temp"].rolling(21).std(),
        "temp_volatility_30": df["temp"].rolling(30).std(),
        "temp_spike_flag": (df["temp"] - df["temp"].shift(1)).abs() > 5,
        "temp_anomaly_vs_month_avg": df["temp"] - df["month_avg_temp"],
        "temp_anomaly_vs_season_avg": df["temp"] - df["season_avg_temp"]
    }
    
    df = df.assign(**volatility_data)

    # FIXED: Pressure trend features
    pressure_data = {
        "pressure_trend_3d": df["sealevelpressure"] - df["sealevelpressure"].shift(3),
        "pressure_trend_7d": df["sealevelpressure"] - df["sealevelpressure"].shift(7),
        "pressure_trend_14d": df["sealevelpressure"] - df["sealevelpressure"].shift(14),
        "pressure_trend_21d": df["sealevelpressure"] - df["sealevelpressure"].shift(21),
        "pressure_trend_30d": df["sealevelpressure"] - df["sealevelpressure"].shift(30)
    }
    
    df = df.assign(**pressure_data)
    df = df.iloc[30:]

    # Return a defragmented copy
    return df.copy()

def select_top_k(X, k=93):
    """Select top K features with error handling"""
    try:
        model_path, selection_path = get_model_paths()
        if selection_path is None or not selection_path.exists():
            print("Selection file not found")
            return X.iloc[:, :k].copy() if len(X.columns) > k else X.copy()
            
        selection_result = joblib.load(selection_path)
        mean_importance = selection_result['importance_mean']
        feature_names = selection_result['feature_names']
        k_features = k

        sorted_idx = mean_importance.argsort()[::-1]
        top_k_idx = sorted_idx[:k_features]
        top_k_features = [feature_names[i] for i in top_k_idx]
        return X[top_k_features].copy()  # Return defragmented copy
    except Exception as e:
        print(f"Feature selection failed: {e}")
        return X.iloc[:, :k].copy() if len(X.columns) > k else X.copy()

def build_preprocessing_pipeline_catboost(X):
    """
    Pipeline TỐI ƯU CHO CATBOOST - FIXED to avoid fragmentation
    """
    # Create a copy to avoid modifying original
    X = X.copy()
    
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    cat_cols.extend(bool_cols)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [col for col in num_cols if col not in bool_cols]

    print(f"Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    print("-" * 50)

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_cols),
            ('cat', 'passthrough', cat_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def predict_future(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Predict future temperatures with comprehensive error handling - FIXED fragmentation
    """
    checking_columns = ['name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
       'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
       'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
       'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
       'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations']
    
    # Input validation
    if df_raw.shape[1] != 33:
        print('The dataframe is not matching! Please recheck columns in the dataframe')
        return pd.DataFrame()  # Return empty DataFrame instead of 0

    if not all(col in df_raw.columns for col in checking_columns):
        print('The dataframe columns do not match expected columns!')
        return pd.DataFrame()

    if df_raw.shape[0] < 30:
        print('There are not enough rows to predict! Need at least 30 rows')
        return pd.DataFrame()

    try:
        # Get model paths
        model_path, selection_path = get_model_paths()
        if model_path is None or not model_path.exists():
            print("Model file not found")
            return pd.DataFrame()
            
        # Load model
        best = joblib.load(model_path)
        
        model = best['model']
        preprocessor = best['preprocessor']
        feature_names = best.get('feature_names', [])
        
        # Process data
        last_date = pd.to_datetime(df_raw['datetime'].iloc[-1])
        X_processed = process_data(df_raw.copy())
        
        # Apply preprocessing
        X_final = preprocessor.transform(X_processed)
        
        # Feature selection
        X_final = select_top_k(X_final)
        
        # Ensure we have the right features for the model
        if hasattr(model, 'feature_names_'):
            missing_features = set(model.feature_names_) - set(X_final.columns)
            if missing_features:
                print(f"Missing features: {missing_features}")
                # Add missing features efficiently
                missing_data = {feature: 0 for feature in missing_features}
                X_final = X_final.assign(**missing_data)
        
        # Predict
        y_pred_5days = model.predict(X_final)[-1]
        y_pred_5days = y_pred_5days[::-1]

        # Create result DataFrame
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='D')
        
        result = pd.DataFrame({
            'date': future_dates,
            'y_pred': y_pred_5days
        })
        
        return result.copy()  # Return defragmented copy
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame()