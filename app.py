import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import subprocess
import numpy as np
import joblib
import threading
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import os
import traceback

import sys
from pathlib import Path
def style_selectboxes():
    st.markdown("""
    <style>
        /* Style the selectbox container */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #000000 !important;
            color: white !important;
        }

        /* Style the selected option */
        .stSelectbox div[data-baseweb="select"] span {
            color: white !important;
        }

        /* Style the dropdown menu items */
        .stSelectbox div[data-baseweb="select"] ul {
            background-color: #000000 !important;
            color: white !important;
        }

        /* Style dropdown options on hover */
        .stSelectbox div[data-baseweb="select"] ul li:hover {
            background-color: #333333 !important;
        }

        /* Ensure text remains readable */
        .stSelectbox div[data-baseweb="select"] input {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
project_root = Path(__file__).parent  
sys.path.append(str(project_root))

def setup_paths() -> Dict[str, Path]:
    """Initialize and validate all file paths"""
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir
        data_dir = current_dir / "data"
        model_dir = current_dir / "models"
        codes_dir = current_dir / "codes"
        
        paths = {
            'data_dir': data_dir,
            'model_dir': model_dir,
            'codes_dir': codes_dir,
            'project_root': project_root,
            'data_file': data_dir / "realtime" / "hanoi_weather_complete.csv",
            'update_script': codes_dir / "update_weather_data.py",
            'model_file': model_dir / "daily" / "BEST_CATBOOST_TIMESERIES.joblib",
            'selection_file': model_dir / "daily" / "selection_result.joblib",
            'prediction_script': codes_dir / "generate_full_multi_horizon_predictions.py"
        }
        
        # Add to Python path safely
        if project_root.exists() and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if codes_dir.exists() and str(codes_dir) not in sys.path:
            sys.path.insert(0, str(codes_dir))
        if model_dir.exists() and str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        if data_dir.exists() and str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))
        return paths
    except Exception as e:
        st.error(f"Path configuration error: {e}")
        # Fallback paths
        current_dir = Path(__file__).parent
        return {
            'data_file': current_dir / "data" / "realtime" / "hanoi_weather_complete.csv",
            'update_script': current_dir / "codes" / "update_weather_data.py",
            'model_file': current_dir / "models" / "daily" / "BEST_CATBOOST_TIMESERIES.joblib",
            'selection_file': current_dir / "models" / "selection_result.joblib"
        }

PATHS = setup_paths()

# --------------------------
# DATA MANAGEMENT
# --------------------------
def is_streamlit_cloud():
    return os.path.exists("/home/appuser")
def safe_prediction_update():
    """Run multi-horizon prediction generation in background"""
    pred_script = PATHS['prediction_script']
    if not pred_script.exists():
        st.sidebar.warning(f"Prediction script not found: {pred_script}")
        return False

    def pred_thread():
        try:
            cmd = [sys.executable, str(pred_script)]
            result = subprocess.run(
                cmd,
                cwd=PATHS['project_root'],
                capture_output=True,
                text=True,
                timeout=60,
                env=os.environ
            )
            if result.returncode == 0:
                print("✅ Multi-horizon predictions updated successfully")
            else:
                print(f"Prediction update failed: {result.stderr}")
        except Exception as e:
            print(f"Prediction thread error: {e}")
            import traceback
            print(traceback.format_exc())

    threading.Thread(target=pred_thread, daemon=True).start()
    st.sidebar.info("Generating multi-horizon predictions...")
    return True
def safe_data_update(run_prediction_after: bool = False):
    if not PATHS['update_script'].exists():
        st.sidebar.warning(f"Update script not found: {PATHS['update_script']}")
        return False

    def update_thread():
        try:
            cmd = [sys.executable, str(PATHS['update_script'])]
            result = subprocess.run(
                cmd,
                cwd=PATHS['project_root'],
                capture_output=True,
                text=True,
                timeout=120,
                env=os.environ
            )
            if result.returncode == 0:
                print("Weather data updated")
                if run_prediction_after:
                    # Now safely run prediction (data is ready)
                    safe_prediction_update()
            else:
                print(f"Update failed: {result.stderr}")
        except Exception as e:
            print(f"Update error: {e}")
            traceback.print_exc()

    threading.Thread(target=update_thread, daemon=True).start()
    st.sidebar.info("Updating weather data...")
    return True

@st.cache_data(ttl=1800)
def load_csv() -> pd.DataFrame:
    """Load weather data with comprehensive error handling"""
    try:
        if not PATHS['data_file'].exists():
            # Create fallback data if file doesn't exist
            st.warning("Data file not found. Using synthetic data.")
            return create_fallback_data()
        
        df = pd.read_csv(PATHS['data_file'], parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        
        # Data quality checks
        if df.empty:
            st.warning("Data file is empty")
            return create_fallback_data()
            
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return create_fallback_data()

def load_historical_forecasts():
    pred_file = PATHS['data_dir'] / "realtime" / "multi_horizon_predictions.csv"
    if not pred_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(pred_file)
        # Force convert columns to datetime, coerce errors to NaT
        df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce')
        df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['as_of_date', 'target_date'])
        return df
    except Exception as e:
        st.error(f"Error loading historical forecasts: {e}")
        return pd.DataFrame()

def create_fallback_data() -> pd.DataFrame:
    """Create synthetic data when real data is unavailable"""
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    data = {
        'datetime': dates,
        'date': [d.date() for d in dates],
        'temp': np.random.normal(27, 3, len(dates)),
        'tempmin': np.random.normal(24, 2, len(dates)),
        'tempmax': np.random.normal(32, 2, len(dates)),
        'conditions': ['Partly Cloudy'] * len(dates),
        'humidity': np.random.normal(75, 10, len(dates)),
        'precip': np.random.normal(10, 5, len(dates)),
        'windspeed': np.random.normal(10, 3, len(dates)),
        'winddir': np.random.normal(180, 90, len(dates)),
        'uvindex': np.random.randint(1, 11, len(dates))
    }
    return pd.DataFrame(data)

def get_historical_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate historical averages for fallback forecasting"""
    try:
        df = df.copy()
        df["doy"] = df["datetime"].dt.dayofyear
        cutoff = pd.Timestamp("today") - pd.Timedelta(days=14)
        hist = df[df["datetime"] < cutoff]
        
        if hist.empty:
            return pd.DataFrame({
                "doy": range(1, 367),
                "tempmin": [24.0] * 366,
                "tempmax": [32.0] * 366
            })
        
        averages = hist.groupby("doy")[["tempmin", "tempmax"]].mean().reset_index()
        full_doy = pd.DataFrame({"doy": range(1, 367)})
        averages = full_doy.merge(averages, on="doy", how="left")
        averages = averages.ffill().bfill()
        return averages
    except Exception:
        # Fallback averages
        return pd.DataFrame({
            "doy": range(1, 367),
            "tempmin": [24.0] * 366,
            "tempmax": [32.0] * 366
        })
def get_weather_predictions(df, today):
    """Get weather predictions with detailed debugging"""
    try:
        # Import here to ensure fresh import with correct paths
        from codes.preprocess_data import predict_future
        
        # Check if model files exist
        model_exists = PATHS['model_file'].exists()
        selection_exists = PATHS['selection_file'].exists()
        
        if not model_exists or not selection_exists:
            st.error(f"Model files not found. Model: {model_exists}, Selection: {selection_exists}")
            return create_fallback_predictions(df, today)
        
        # Check if dataframe has required columns
        required_cols = ['name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
                        'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
                        'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
                        'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                        'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
                        'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return create_fallback_predictions(df, today)
        
        if len(df) < 30:
            st.error(f"Not enough data rows. Need 30, have {len(df)}")
            return create_fallback_predictions(df, today)
        
        df = df.sort_values("datetime").reset_index(drop=True)
        predict_results = predict_future(df[-60:])
        
        if isinstance(predict_results, pd.DataFrame):
            if not predict_results.empty:
                if 'y_pred' in predict_results.columns and 'date' in predict_results.columns:
                    predict_results['temp'] = predict_results['y_pred'].astype(float)
                    predict_results['date'] = pd.to_datetime(predict_results['date'])
                    return predict_results[['date', 'temp']]
                else:
                    st.error("Prediction results missing required columns")
            else:
                st.error("Prediction returned empty DataFrame")
        else:
            st.error(f"Prediction failed with return code: {predict_results}")
            
        return create_fallback_predictions(df, today)

    except Exception as e:
        st.error(f" Error during prediction: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return create_fallback_predictions(df, today)

def create_fallback_predictions(df, today):
    """Create robust fallback predictions when model fails"""
    try:
        averages = get_historical_averages(df)
        future_dates = [today + timedelta(days=i+1) for i in range(5)]
        
        predictions = []
        for i, target_date in enumerate(future_dates):
            doy = target_date.timetuple().tm_yday
            
            # Get historical average for this day of year
            avg_row = averages[averages['doy'] == doy]
            
            if not avg_row.empty:
                temp = (avg_row['tempmin'].iloc[0] + avg_row['tempmax'].iloc[0]) / 2
            else:
                # Use overall average if specific day not found
                temp = 27.0
            
            predictions.append({
                "date": pd.Timestamp(target_date),  # Ensure datetime type
                "temp": temp
            })
        
        return pd.DataFrame(predictions)
        
    except Exception as e:
        st.error(f"Fallback prediction also failed: {e}")
        # Ultimate fallback - simple predictions
        future_dates = [today + timedelta(days=i+1) for i in range(5)]
        return pd.DataFrame({
            "date": [pd.Timestamp(d) for d in future_dates],  # Ensure datetime
            "temp": [27.0, 27.5, 26.8, 28.0, 27.2]
        })


# --------------------------
# STREAMLIT UI SETUP
# --------------------------
def setup_page():
    """Configure Streamlit page and styles"""

    st.set_page_config(
        page_title="Hanoi Weather Forecast",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )
    st.markdown("""
    <style>
    /* MAIN APP - White background */
    .stApp {
        background: #f9faff !important;
        color: #1e293b !important;
        zoom: 1;              /* For Chrome/Edge */
        transform: scale(1);  /* For Firefox and others */
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    .stSelectbox, .stDateInput, .stSelectbox label, .stDateInput label, .stSelectbox div, .stDateInput div {
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #575ea5 !important;  /* Purple background */
        color: #ffffff !important;             /* White text */
        border-radius: 15px !important;
        padding: 12px 24px !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        background-color: #79a8e7 !important;
        color: #182850 !important;
        transform: translateY(-2px);
    }

    .stButton>button:active {
        transform: translateY(1px);
    }
    /* REMOVE ALL TOP/BOTTOM PADDING */
    .main .block-container {padding:0rem !important;}

    .stDateInput input {
        color: black !important;
        background-color: white !important;
    }
    .stDateInput label {
        color: black !important;
    }
    /* Style the temperature trend chart background */
    div[data-testid="stChart"] {
        background-color: white !important;
        padding: 0.1rem !important;
        border-radius: 12px !important;
    }               
    .temperature-trend-container {
        background-color: #ffffff !important;
    }

    .temperature-trend-container div[data-testid="stChart"] {
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;     /* White background */
        color: #000000 !important;                /* Black text */
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    /* Also fix the SVG/chart internals if needed */
    .stLineChart svg {
        background-color: #ffffff !important;
    }         
    /* WIDER SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #2e3190  !important;
        min-width: 280 !important;
        max-width: 300px !important;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .sidebar-title {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: white !important;
        margin-bottom: 1rem !important;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background: #575ea5 !important;
        color: #ffffff !important;
        border-radius: 15px;
        width: 90%;
        padding: 12px;
        margin: 3px 0;
        font-size: 0.5rem;
        font-weight: 500;
        border: 0px;
        transition: all 0.2s;
    }
    
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: #79a8e7  !important;
        color: #182850  !important;
        transform: translateX(4px);
    }
    
    section[data-testid="stSidebar"] .stButton>button[aria-pressed="true"] {
        background: #79a8e7  !important;
        color: #ffffff !important;
        font-weight: bold;
    }                        
    /* TITLES - Left aligned */
    .left-title {
        text-align: left !important;
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding: 0rem 0rem !important;       
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    .left-date {
        text-align: left !important;
        color: #000000 !important;
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding: 0.2rem 0rem !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
    }

    /* SINGLE CONTAINER FOR ALL CURRENT WEATHER */
    .current-weather-full-container {
        background: #2e3190;
        color: #ffffff;
        border-radius: 16px;
        padding: 0rem 1rem;
        margin: 0rem auto 0rem auto; 
        width: 100%; /* Stretch to full width */
        max-width: none; /* Remove max-width constraint */
        box-shadow: 0 10px 17px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        align-items: center; 
        height: 28vh;
    }

    .current-weather-heading {
        text-align: right;
        color: white;
        margin: 0 0 0.2rem 0;
        font-size: 1.2rem;
        font-weight: 550;
        padding: 0;
    }
                
    .weather-header {
        display: flex;
        align-items: center; /* Keep everything vertically centered */
        gap: 0.3rem; /* Reduced from 2rem/3rem to tighten space */
        margin-top: 0rem;
        margin-bottom: 0rem;
        padding: 0rem 0rem;
        flex-wrap: wrap; /* Allow wrapping on small screens */
        justify-content: center; /* Center the header content */
        text-align: center; 
    }

    .weather-icon-temp {
        display: flex;
        align-items: center;
        gap: 0.7rem; /* slightly more breathing room */
    }
    
    .weather-icon {
        font-size: 3.2rem !important;
    }
    
    .weather-temp {
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: #ffffff !important;
        text-align: left;
    }
    
    .weather-condition {
        text-align: left;
        font-size: 0.7rem !important;
        color: #ffffff !important;
    }
    
    .weather-minmax {
        font-size: 0.7rem !important;
        color: #ffffff !important;
        text-align: right;
    }

    /* METRICS ROW - inside the same container */
    .weather-metrics-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 7rem;
    }
    
    .weather-metric-item {
        text-align: center;
        min-width: 80px;
        line-height: 1; 
    }
    
    .metric-value {
        font-size: 0.7rem;
        font-weight: bold;
        color: ffffff;
    }
    
    .metric-label {
        font-size: 0.6rem;
        color: #ffffff;
        margin-top: 0.25rem;
    }


    /* FORECAST CARDS */
    .forecast-card {
        background: transparent !important; /* or remove background property */
        box-shadow: 0 10px 17px rgba(0,0,0,0.1);
        padding-top: 0.6rem;
        padding-bottom: 0rem;
        border-radius: 15px;
        text-align: center;
        height: 100%;
    }

    /* Style the parent div that Streamlit creates */
    div[data-testid="column"] > div:nth-child(1) {
        background: #ffffff !important;
        border-radius: 15px;
        padding: 0.6rem 0 !important;
        height: 100%;
    }
        
    .forecast-card .icon {
        font-size: 5rem !important;
        margin: 0rem 0;
    }
    
    .forecast-card .temp {
        font-size: 2rem !important;
        font-weight: bold !important;
        padding = 0rem !important;
    }
    
    /* Remove table borders */
    .current-weather-table, .current-weather-table td {
        border: none !important;
        background: transparent !important;
    }
    .forecast-card-table, .forecast-card-table td {
        border: none !important;
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# RENDERING FUNCTIONS - UPDATED LAYOUT
# --------------------------
def render_forecasting(df, today):
    """Render main forecasting page with updated layout"""
    try:
        # Title and date on the LEFT
        st.markdown(f"<h1 class='left-title' style='margin:0.1rem;'>HANOI WEATHER FORECAST</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='left-date'>{today.strftime('%d/%m/%Y')}</h3>", unsafe_allow_html=True)

        # Get model predictions or fallback
        future_df = get_weather_predictions(df, today)
        
        # Display current weather - CENTERED
        render_current_weather(df, today)
                
        # Display forecast
        render_forecast_cards(df, today, future_df)
        
        # Display temperature trend
        historical_pred_df = load_historical_forecasts()  # your existing loader

        # Render the unified chart
        render_forecast_comparison(df, historical_pred_df, today)
        
    except Exception as e:
        st.error(f"Forecast rendering error: {e}")
        # Provide more detailed error information
        import traceback
        st.error(f"Full error details: {traceback.format_exc()}")
        render_fallback_forecast(df, today)

def render_current_weather(df, today):
    """Render current weather section - ALL in one container"""
    try:
        # Ensure we have a date column
        if 'date' not in df.columns and 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
        
        row_today = df[df["date"] == today]
        if not row_today.empty:
            r = row_today.iloc[0]
            temp = float(r.get("temp", 27))
            tmax = float(r.get("tempmax", 30))
            tmin = float(r.get("tempmin", 23))
            cond = str(r.get("conditions", "Rainy")).split(",")[0]

            # Weather icon mapping
            icon_map = {
                "Clear": "☀️", "Partly Cloudy": "🌤️", "Cloudy": "☁️", "Light Rain": "🌧️",
                "Rain": "🌧️", "Heavy Rain": "⛈️", "Snow": "❄️", "Thunderstorm": "⚡",
                "Fog": "🌫️", "Windy": "💨", "Mostly Cloudy": "⛅", "Rainy": "🌧️",
                "Partly cloudy": "🌤️", "Overcast": "☁️", "Sunny": "☀️"
            }
            icon = icon_map.get(cond, "🌤️")

            # SINGLE CONTAINER: icon, temp, min/max, AND metrics
            st.markdown(
                f"""
                <div class="current-weather-full-container">
                    <div class="current-weather-heading" margin-top:0rem; font-size:1.5rem; color: #ffffff; text-align: right;'>Current Weather</div>
                    <div class="weather-header">
                        <div class="weather-icon-temp">
                            <div class="weather-icon">{icon}</div>
                            <div>
                                <div class="weather-temp">{temp:.0f}°C</div>
                                <div class="weather-condition" style="display:flex; align-items:center;">
                                    {cond}
                                </div>
                            </div>
                        </div>
                        <div class="weather-minmax">
                            Max: {tmax:.0f}°C<br>Min: {tmin:.0f}°C
                        </div>
                    </div>
                    <div class="weather-metrics-row">
                        <div class="weather-metric-item">
                            <div class="metric-value">💧 {r.get('humidity', 33):.0f}%</div>
                            <div class="metric-label">Humidity</div>
                        </div>
                        <div class="weather-metric-item">
                            <div class="metric-value">🌧️ {r.get('precip', 34):.0f}%</div>
                            <div class="metric-label">Precip</div>
                        </div>
                        <div class="weather-metric-item">
                            <div class="metric-value">🧭 {r.get('winddir', 67):.0f}°</div>
                            <div class="metric-label">Wind direction</div>
                        </div>
                        <div class="weather-metric-item">
                            <div class="metric-value">🌬️ {r.get('windspeed', 67):.1f}</div>
                            <div class="metric-label">Wind speed</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("No current weather data available")
    except Exception as e:
        st.error(f"Error rendering current weather: {e}")

def render_forecast_cards(df, today, future_df):
    """Render forecast cards with better error handling."""
    st.markdown("<h3 style='text-align:left; margin-top:0rem; font-size:1rem;'>Next 5 Days Temperature Forecast</h3>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    
    # Ensure we have a valid DataFrame with proper columns
    if future_df is None or future_df.empty:
        st.warning("No forecast data available. Using fallback predictions.")
        future_df = create_fallback_predictions(df, today)
    
    # Ensure required columns exist with proper data types
    if 'date' not in future_df.columns:
        future_df['date'] = [pd.Timestamp(today + timedelta(days=i+1)) for i in range(len(future_df))]
    
    if 'temp' not in future_df.columns:
        future_df['temp'] = 27.0  # Default temperature
    
    # Ensure dates are proper datetime objects
    try:
        future_df['date'] = pd.to_datetime(future_df['date'])
    except Exception as e:
        st.warning(f"Date conversion issue: {e}")
        # Create proper dates as fallback
        future_df['date'] = pd.date_range(today + timedelta(days=1), periods=len(future_df), freq="D")
    
    # Render the cards
    for i in range(5):
        target_date = today + timedelta(days=i + 1)
        day_abbr = target_date.strftime("%a")
        date_str = target_date.strftime("%d/%m")
        
        # Find matching row - ensure we compare date parts correctly
        try:
            matching_rows = future_df[future_df['date'].dt.date == target_date]
        except Exception:
            # If date comparison fails, use index-based approach
            if i < len(future_df):
                matching_rows = future_df.iloc[[i]]
            else:
                matching_rows = pd.DataFrame()
        
        with cols[i]:
            if not matching_rows.empty and len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                temp = float(row.get('temp', 27.0))
                
                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        padding:0.3rem 1.0rem;
                        border-radius:14px;
                        color: 1e293b;
                        width:100%;
                        display:flex;
                        justify-content:space-between;
                        box-shadow: 0 10px 17px rgba(0,0,0,0.1);
                        align-items:center;
                    ">
                        <div style="text-align:left;">
                            <div style="font-size:0.7rem; font-weight:600;">{day_abbr}</div>
                            <div style="font-size:0.7rem; opacity:0.85;">{date_str}</div>
                        </div>
                        <div style="text-align:right; font-size:1.3rem; font-weight:600;">
                            {temp:.0f}°C
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Fallback card when no data is available
                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        padding:1.2rem 1.0rem;
                        border-radius:14px;
                        color:203a7d;
                        width:100%;
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                    ">
                        <div style="text-align:left;">
                            <div style="font-size:1.5rem; font-weight:600;">{day_abbr}</div>
                            <div style="font-size:1.2rem; opacity:0.7;">{date_str}</div>
                        </div>
                        <div style="text-align:right; font-size:2rem; opacity:0.4; font-weight:700;">
                            -
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def render_forecast_comparison(df, historical_pred_df, today):
    """
    Show actual temps + 5 horizon lines (1-day to 5-day forecasts)
    from historical predictions (dates ≤ today).
    """
    try:
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)  # space before chart
        today_ts = pd.Timestamp(today)

        # --- Actual temperatures ---
        actual = df[['datetime', 'temp']].copy()
        actual['datetime'] = pd.to_datetime(actual['datetime'])
        actual = actual.set_index('datetime').sort_index()
        start_date = historical_pred_df["target_date"].min()
        actual = actual[(actual.index <= today_ts) & (actual.index >= start_date)]

        # --- Initialize Plotly figure ---
        fig = go.Figure()

        # >>> ACTUAL LINE (SOLID) <<<
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual['temp'],
            mode='lines',
            name='Actual',
            line=dict(color="#3a0eca", width=2.5)  # Dark gray/black
        ))

        # --- 5 HORIZON LINES (DOTTED) ---
        if not historical_pred_df.empty:
            past_pred = historical_pred_df[
                historical_pred_df['as_of_date'] <= today_ts
            ].copy()

            if not past_pred.empty:
                past_pred = past_pred.sort_values('as_of_date')
                # Color palette: red → blue (1-day to 5-day)
                colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
                
                for horizon in [1, 2, 3, 4, 5]:
                    h_data = past_pred[past_pred['horizon'] == horizon]
                    if not h_data.empty:
                        # Get most recent prediction for each target_date
                        h_series = h_data.groupby('target_date')['predicted_temp'].last()
                        
                        fig.add_trace(go.Scatter(
                            x=h_series.index,
                            y=h_series.values,
                            mode='lines',
                            name=f'{horizon}-Day Forecast',
                            line=dict(color=colors[horizon-1], width=1.4, dash='dot')
                        ))
        # --- Add TODAY vertical line ---
        fig.add_vline(
            x=today_ts,
            line_width=1.4,
            line_dash="dash",
            line_color="red"
        )
        fig.add_annotation(
            x=today_ts,
            y=actual['temp'].max(),
            text="Today",
            showarrow=False,
            font=dict(color="black", size=10),
            yshift=10
        )

        # --- Layout ---
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Date",
                    font=dict(color="black", size=10)  # <-- correct way
                ),
                tickfont=dict(color="black", size=8),
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title=dict(
                    text="Temperature (°C)",
                    font=dict(color="black", size=10)  # <-- correct
                ),
                tickfont=dict(color="black", size=8),
                gridcolor='#e2e8f0',
                range=[0, actual['temp'].max() + 2] 
            ),
            plot_bgcolor='white',     # INSIDE background
            paper_bgcolor='white',    # OUTSIDE background
            height=150,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="black", size=10)
            ),
            margin = dict(t=0, b=0, l=5, r=5)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e}")
        import traceback
        st.text(traceback.format_exc())


def render_fallback_forecast(df, today):
    """Render fallback forecast when main rendering fails"""
    st.warning("Displaying simplified forecast due to technical issues")
    
    try:
        # Simple current weather display
        if 'date' not in df.columns and 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
            
        row_today = df[df["date"] == today]
        if not row_today.empty:
            r = row_today.iloc[0]
            st.metric("Current Temperature", f"{r.get('temp', 27):.1f}°C")
        
        # Simple forecast
        st.subheader("Next 5 Days (Estimate)")
        for i in range(5):
            target_date = today + timedelta(days=i + 1)
            st.write(f"{target_date.strftime('%a')}: ~27°C")
    except Exception as e:
        st.error(f"Fallback forecast also failed: {e}")

def render_past_weather(df):
    st.markdown("<h1 style='text-align:center;font-size: 1.5rem;'>HANOI WEATHER HISTORY</h1>", unsafe_allow_html=True)

    # Ensure 'date' column exists and is of type date
    if 'date' not in df.columns:
        if 'datetime' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['datetime']).dt.date
        else:
            st.error("DataFrame must contain a 'date' or 'datetime' column.")
            return
    else:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.date

    min_date = df['date'].min()
    max_date = min(df['date'].max(), datetime.now().date())

    # === Mode Selection (Styled to appear on dark background) ===
    # Optional: wrap in a dark-background container for visual emphasis
    st.markdown('<span style="background-color:#ffffff; color:black; font-size:1rem;">Select mode</span>', unsafe_allow_html=True)
    mode = st.selectbox(
        "Select mode",  # accessibility fallback
        options=["Single Date", "Date Range"],
        index=0,
        key="mode_select",
        label_visibility="collapsed"
    )
   
    if mode == "Single Date":
        search_date = st.date_input(
            "Select a date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="single_date"
        )
        # Auto-update without button
        row = df[df['date'] == search_date]
        if not row.empty:
            st.markdown(f"##### Weather on {search_date.strftime('%b %d, %Y')}")
            st.dataframe(row, use_container_width=True)
        else:
            st.warning(f"No data found for {search_date}")

    elif mode == "Date Range":
    # Top row: Start, End, View Mode
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            start_date = st.date_input(
                "Start date",
                value=max_date - timedelta(days=6),
                min_value=min_date,
                max_value=max_date,
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date"
            )
        with col3:
            view_mode = st.selectbox(
                "View",
                options=["Single Feature", "All Features"],
                index=0,
                key="view_mode"
            )

        if start_date > end_date:
            st.warning("Start date cannot be after end date.")
            st.stop()

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        range_df = df.loc[mask].copy()

        if range_df.empty:
            st.warning(f"No data found between {start_date} and {end_date}.")
            st.stop()

        # Shared setup
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        preferred_order = ['temp', 'tempmax', 'tempmin', 'humidity', 'precip', 'windspeed', 'windgust', 'pressure', 'uvindex']
        numeric_cols_sorted = sorted(
            numeric_cols,
            key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x)
        )
        unit_map = {
            'temp': '°C', 'tempmax': '°C', 'tempmin': '°C',
            'humidity': '%', 'precip': 'mm', 'windspeed': 'km/h',
            'windgust': 'km/h', 'pressure': 'hPa', 'uvindex': ''
        }

        # Helper: render one feature
        def render_feature(feature):
            feature_series = pd.to_numeric(range_df[feature], errors='coerce').dropna()
            if feature_series.empty:
                return False

            stats = {
                'Min': feature_series.min(),
                'Max': feature_series.max(),
                'Mean': feature_series.mean(),
                'Std Dev': feature_series.std(),
                'Count': int(feature_series.count())
            }
            unit = unit_map.get(feature, "")

            chart_df = range_df[['date', feature]].dropna().sort_values('date')
            chart_df['date_str'] = chart_df['date'].apply(lambda d: d.strftime('%b %d'))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_df['date_str'],
                y=chart_df[feature],
                mode='lines+markers',
                line=dict(color="#2a12a3", width=2.2),
                marker=dict(size=5),
                hovertemplate=f'%{{x}}<br>%{{y:.1f}}{unit}<extra></extra>'
            ))
            fig.update_layout(
                title=f"{feature.replace('_',' ').title()} ({unit})",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                height=200,
                xaxis=dict(gridcolor='#e2e8f0', tickfont=dict(size=10, color="black")),
                yaxis=dict(title=unit, gridcolor='#e2e8f0', tickfont=dict(size=10, color="black")),
                showlegend=False
            )

            chart_col, stats_col = st.columns([3, 1])
            with chart_col:
                st.plotly_chart(fig, use_container_width=True)
            with stats_col:
                st.markdown("##### Summary")
                for label, value in stats.items():
                    if pd.isna(value):
                        display_val = "N/A"
                    elif label == 'Count':
                        display_val = str(int(value))
                    else:
                        display_val = f"{value:.1f}{unit}"
                    color = '#ef4444' if label == 'Max' else '#3b82f6' if label == 'Min' else '#203a7d'
                    st.markdown(f"<div style='font-size: 0.9rem; line-height: 1.5; margin-bottom: 0.5rem;'><strong>  {label}:</strong> <span style='color:{color};'>{display_val}</span></div>", unsafe_allow_html=True)
            return True

        if view_mode == "Single Feature":
            # Show feature selector below the top row
            selected_feature = st.selectbox(
                "Feature to analyze",
                options=numeric_cols_sorted,
                index=numeric_cols_sorted.index('temp') if 'temp' in numeric_cols_sorted else 0,
                key="feature_select_single"
            )
            render_feature(selected_feature)
            with st.expander("View Raw Data", expanded=False):
                display_df = range_df.copy()
                display_df['date'] = display_df['date'].apply(lambda d: d.strftime('%Y-%m-%d'))
                cols_to_show = ['date', selected_feature]
                if 'conditions' in display_df.columns:
                    cols_to_show.append('conditions')
                st.dataframe(display_df[cols_to_show].sort_values('date', ascending=False), use_container_width=True)
        else:  # "All Features"
            st.markdown("###### All Features")
            with st.expander("View Full Data (All Columns)", expanded=False):
                display_df = range_df.copy()
                display_df['date'] = display_df['date'].apply(lambda d: d.strftime('%Y-%m-%d'))
                display_df = display_df.sort_values('date', ascending=False)
                st.dataframe(display_df, use_container_width=True)
def render_model_performance():
    """Render model performance page with metric cards"""
    st.markdown("<h1 style='text-align:center; font-size: 1.5rem;'>MODEL PERFORMANCE</h1>", unsafe_allow_html=True)
    st.markdown("###### Performance Metrics")
    st.markdown(
        """
        <div style="
            background: #ffffff;
            padding: 0.5rem 3rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            text-align: left;
        ">
            <div style="display: flex; gap: 10rem;">
                <div style="font-size: 0.8rem; line-height: 1.7;">
                    <div>Model:</div>
                    <div>Input Features: </div>
                    <div>Training Period: </div>
                    <div>Prediction Target: </div>
                </div>
                <div style="font-size: 0.8rem; line-height: 1.7; text-align: left;">
                    <div> <strong>CatBoost</strong></div>
                    <div> <strong>1200 features</strong></div>
                    <div> <strong>01/10/2015–01/10/2025</strong></div>
                    <div><strong>Daily average temperature</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("###### Performance Metrics")
    
    # Define your metrics
    metrics = [
        ("R² Score", "0.82"),
        ("MAE", "1.68°C"),
        ("RMSE", "2.15°C"),
        ("Training Period", "9 years")
    ]
    
    # Create 2 columns
    cols = st.columns(2)
    
    for i, (label, value) in enumerate(metrics):
        with cols[i % 2]:  # Alternate between col0 and col1
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    padding:0.6rem 0.7rem;
                    margin: 0.3rem 0.5rem;
                    border-radius:12px;
                    color:black;
                    width:100%;
                    display:flex;
                    flex-direction:row;
                    align-items:center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    justify-content:space-between;
                ">
                    <div style="text-align:left; font-size:0.8rem; opacity:0.8;">{label}</div>
                    <div style="text-align:center; font-size:0.8rem; font-weight:600;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("###### Feature Importance")
    st.markdown("""
    <div style="
        font-size: 0.8rem;
        background:#dbeafe;
        color:#0c4a6e;
        padding:8px 10px;
        border-left:4px solid #3b82f6;
        border-radius:5px;
    ">
    <b>Top 10 features:</b> day_length_hours_lag_21, day_length_hours_lag_30,
    temp_sealevelpressure_interaction, feelslike, temp, day_avg_feelslike,
    day_avg_tempmin, rolling_30_sealevelpressure, rolling_3_sealevelpressure_change,
    season_avg_sealevelpressure
    </div>
    """, unsafe_allow_html=True)

def render_other_settings():
    """Render settings page with black text"""
    # Main heading in black
    st.markdown("<h1 style='text-align:center; color:#000000;font-size: 1.5rem;'>OTHER SETTINGS</h1>", unsafe_allow_html=True)
    
    # Subheading in black
    st.markdown("<h4 style='color:#000000;'>Data Management</h4>", unsafe_allow_html=True)
    
    if st.button("Update Weather Data Now"):
        with st.spinner("Updating weather data..."):
            if safe_data_update():
                st.success("Update started in background")
                st.cache_data.clear()
            else:
                st.error("Update failed")

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully")

# --------------------------
# MAIN APPLICATION
# --------------------------
def main():
    """Main application entry point with comprehensive error handling."""
    try:
        # Setup page configuration
        setup_page()
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = "Forecasting"
        
        # Load data with error handling
        try:
            with st.spinner("Loading weather data..."):
                df = load_csv()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            df = create_fallback_data()
        
        # Get current date
        today = datetime.now().date()

        # Auto-update
        if not st.session_state.get('auto_update_done', False):
            if is_streamlit_cloud():
                st.session_state.auto_update_done = True  # skip
            else:
                try:
                    if not df.empty and 'datetime' in df.columns:
                        last_recorded = pd.to_datetime(df['datetime']).dt.date.max()
                        if today > last_recorded:
                            st.sidebar.info("Data is outdated. Updating...")
                            st.cache_data.clear()
                            safe_data_update(run_prediction_after=True)  
                            st.session_state.auto_update_done = True
                        else:
                            st.session_state.auto_update_done = True
                    else:
                        st.session_state.auto_update_done = True
                except Exception as e:
                    st.sidebar.error(f"Auto-update failed: {e}")
                    st.session_state.auto_update_done = True
        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-title">Navigation Menu</div>', unsafe_allow_html=True)
            
            pages = {
                "Forecasting": "Weather Forecast",
                "Past weather data": "Historical Data", 
                "Model performance": "Model Info",
                "Other settings": "Settings"
            }
            
            for page_key, page_label in pages.items():
                if st.button(page_label, width ="stretch", key=page_key):
                    st.session_state.page = page_key
                    st.rerun()
            
            st.markdown("---")
            st.markdown("###### Data Status")
            
            # Show data freshness
            try:
                if not df.empty:
                    # Ensure we have datetime column for last update
                    if 'datetime' in df.columns:
                        last_date = df["datetime"].max()
                        if hasattr(last_date, 'strftime'):
                            last_update = last_date.strftime("%b %d, %Y")
                        else:
                            last_update = "Unknown"
                    else:
                        last_update = "Unknown"
                        
                    st.caption(f"Last update: {last_update}")
                    st.caption(f"Records: {len(df):,}")
                else:
                    st.caption("Data status: No data available")
            except Exception:
                st.caption("Data status: Unknown")
        
        # Page routing with error handling
        try:
            if st.session_state.page == "Forecasting":
                render_forecasting(df, today)
            elif st.session_state.page == "Past weather data":
                render_past_weather(df)
            elif st.session_state.page == "Model performance":
                render_model_performance()
            elif st.session_state.page == "Other settings":
                render_other_settings()
        except Exception as e:
            st.error(f"Page rendering error: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.info("Please try refreshing the page or selecting a different section.")
        
        # Footer
        st.markdown("---")
        st.caption("Hanoi Weather Forecast • Data: Visual Crossing Weather API • Built with Streamlit")
        
    except Exception as e:
        st.error(f"Critical application error: {e}")
        import traceback
        st.error(f"Full error details: {traceback.format_exc()}")
        st.info("The application has encountered a critical error. Please refresh the page.")

if __name__ == "__main__":
    main()
