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
        model_file = project_root / "models" / "daily" / "BEST_CATBOOST_TUNED_DAILY.joblib"
        selection_file = project_root / "models" / "daily" / "selection_result_daily.joblib"
        
        print(f"Looking for model in: {model_file}")
        print(f"Model exists: {model_file.exists()}")
        print(f"Selection exists: {selection_file.exists()}")
        
        return model_file, selection_file
    except Exception as e:
        print(f"Error getting model paths: {e}")
        return None, None


def process_data(df: pd.DataFrame):
    """
    Process raw weather data (33 columns) into engineered features for prediction.
    Input: DataFrame with raw weather columns (no targets).
    Output: Engineered feature DataFrame (X only), indexed by datetime.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Extract time components
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_year'] = df.index.dayofyear

    # Season mapping
    season_mapping = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['season'] = df['month'].map(season_mapping)

    # Drop 'conditions' — explicitly removed in training
    if 'conditions' in df.columns:
        df = df.drop(columns=['conditions'])

    # Handle missing values (match preprocessing step)
    df['preciptype'] = df['preciptype'].fillna('none')
    df['severerisk'] = df['severerisk'].fillna(0)

    # Temperature range
    df['temp_range'] = df['tempmax'] - df['tempmin']

    # Day length
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['day_length_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600.0
    df.drop(columns=['sunrise', 'sunset'], inplace=True)

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Interaction features
    df["temp_solar_interaction"] = df["temp"] * df["solarradiation"]
    df["uv_temp_interaction"] = df["uvindex"] * df["temp"]
    df['temp_cloudcover_interaction'] = df['temp'] * df['cloudcover']
    df['temp_sealevelpressure_interaction'] = df['temp'] * df['sealevelpressure']
    df['weighted_precip'] = df['precipprob'] * df['precip']
    df['effective_solar'] = df['solarradiation'] * (1 - df['cloudcover'] / 100)
    df['precip_impact'] = df['precipprob'] * df['precip']
    df['temp_minus_dew'] = df['temp'] - df['dew']

    # Wind vector decomposition
    df['wind_u'] = df['windspeed'] * np.cos(2 * np.pi * df['winddir'] / 360)
    df['wind_v'] = df['windspeed'] * np.sin(2 * np.pi * df['winddir'] / 360)
    df.drop('winddir', axis=1, inplace=True)

    # Moon phase encoding
    df['moonphase_sin'] = np.sin(2 * np.pi * df['moonphase'])
    df['moonphase_cos'] = np.cos(2 * np.pi * df['moonphase'])
    df.drop('moonphase', axis=1, inplace=True)

    # ✅ Define computing_columns AFTER all base features are created
    exclude_cols = {
        'year', 'month', 'day', 'day_of_year',
        'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
        'season', 'preciptype', 'name', 'description', 'icon', 'stations'
    }
    computing_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    # Lag features
    lag_steps = [1, 2, 3, 5, 7, 10, 14, 21, 30]
    for col in computing_columns:
        for lag in lag_steps:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Rolling features
    rolling_horizons = [3, 4, 5, 7, 14, 21, 30]
    for horizon in rolling_horizons:
        for col in computing_columns:
            rolling_mean = df[col].rolling(window=horizon, min_periods=horizon).mean()
            df[f"rolling_{horizon}_{col}"] = rolling_mean
            df[f"rolling_{horizon}_{col}_change"] = df[col] - rolling_mean

    # Expanding (cumulative) averages
    def expand_mean(series):
        return series.expanding(1).mean()
    
    for col in computing_columns:
        df[f"month_avg_{col}"] = df.groupby('month')[col].transform(lambda x: x.expanding().mean())
        df[f"day_avg_{col}"] = df.groupby(df.index.day_of_year)[col].transform(lambda x: x.expanding().mean())
        df[f"year_avg_{col}"] = df.groupby('year')[col].transform(lambda x: x.expanding().mean())
        df[f"season_avg_{col}"] = df.groupby('season')[col].transform(lambda x: x.expanding().mean())

    # Cumulative max/min by day (for temp only)
    df["day_max_temp"] = df.groupby(df.index.dayofyear)['temp'].cummax()
    df["day_min_temp"] = df.groupby(df.index.dayofyear)['temp'].cummin()

    # Anomalies
    df["temp_anomaly_vs_month_avg"] = df["temp"] - df["month_avg_temp"]
    df["temp_anomaly_vs_day_avg"] = df["temp"] - df["day_avg_temp"]
    df["temp_anomaly_vs_season_avg"] = df["temp"] - df["season_avg_temp"]

    # Pressure trends (differences)
    lags_pressure = [2, 3, 5, 7, 14, 21, 30]
    for lag in lags_pressure:
        df[f"pressure_trend_{lag}d"] = df["sealevelpressure"].diff(lag)

    # Final cleanup: remove raw time columns
    df.drop(columns=['day', 'month', 'day_of_year'], inplace=True, errors='ignore')

    # Remove first 30 rows to handle max lag/rolling window
    df = df.iloc[30:].copy()
        # Keep only numerical features (critical for alignment with training)
    df = df.select_dtypes(include=[np.number])
    return df


def select_top_k(X, k=95):
    """Select top K features using saved selection result."""
    try:
        _, selection_path = get_model_paths()
        if not selection_path or not selection_path.exists():
            print("Selection file not found — using first K columns")
            return X.iloc[:, :k].copy() if X.shape[1] > k else X.copy()
            
        selection_result = joblib.load(selection_path)
        mean_importance = selection_result['importance_mean']
        feature_names = selection_result['feature_names']
        sorted_idx = mean_importance.argsort()[::-1]
        top_k_features = [feature_names[i] for i in sorted_idx[:k]]
        
        # Safely select available features
        available_features = [f for f in top_k_features if f in X.columns]
        missing = set(top_k_features) - set(available_features)
        if missing:
            print(f"Warning: {len(missing)} top features missing, using {len(available_features)} available")
        return X[available_features].copy()
    except Exception as e:
        print(f"Feature selection failed: {e}")
        return X.iloc[:, :k].copy() if X.shape[1] > k else X.copy()


def predict_future(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Predict next 5 days of temperature using trained CatBoost model.
    Assumes model was saved as: {'model': catboost_model, 'feature_names': [...]}
    """
    required_columns = [
        'name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
        'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
        'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
        'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
        'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
        'sunset', 'moonphase', 'description', 'icon', 'stations'
        # 'conditions' is intentionally excluded — dropped during training
    ]
    
    # Input validation
    if df_raw.shape[1] != 33:
        print('❌ The dataframe does not have 33 columns!')
        return pd.DataFrame()

    if not all(col in df_raw.columns for col in required_columns):
        print('❌ Missing required columns!')
        return pd.DataFrame()

    if df_raw.shape[0] < 30:
        print('❌ Need at least 30 historical rows for feature engineering!')
        return pd.DataFrame()

    try:
        model_path, _ = get_model_paths()
        if not model_path or not model_path.exists():
            print("❌ Model file not found!")
            return pd.DataFrame()

        # Load model package
        model_package = joblib.load(model_path)
        model = model_package['model']
        expected_features = model_package.get('feature_names', None)

        if expected_features is None:
            print("⚠️ Warning: 'feature_names' not in model — using all columns after selection")
        
        # Process raw data
        last_date = pd.to_datetime(df_raw['datetime'].iloc[-1])
        X_processed = process_data(df_raw.copy())

        # Apply feature selection
        X_selected = select_top_k(X_processed)

        # Align with model's expected features
        if expected_features:
            missing_features = set(expected_features) - set(X_selected.columns)
            if missing_features:
                print(f"⚠️ Adding {len(missing_features)} missing features as 0")
                for feat in missing_features:
                    X_selected[feat] = 0
            # Reorder to match training order
            X_final = X_selected.reindex(columns=expected_features, fill_value=0)
        else:
            X_final = X_selected

        # Predict — output is already [T+1, T+2, T+3, T+4, T+5]
        y_pred = model.predict(X_final)
        y_pred_5days = y_pred[-1]  # shape: (5,)

        # Create result
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='D')
        result = pd.DataFrame({
            'date': future_dates,
            'y_pred': y_pred_5days
        })

        return result.copy()

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    