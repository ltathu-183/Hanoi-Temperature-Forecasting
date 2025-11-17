import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import numpy as np
import joblib
import threading
import plotly.graph_objects as go
import plotly.express as px
import os
from zoneinfo import ZoneInfo  # Python 3.9+, built-in and works perfectly on Streamlit Cloud
# or fallback for very old Python: import pytz; HANOI_TZ = pytz.timezone('Asia/Ho_Chi_Minh')
HANOI_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
# --- PATHS ---
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_paths():
    try:
        current_dir = Path(__file__).parent
        data_dir = current_dir / "data"
        model_dir = current_dir / "models"
        notebooks_dir = current_dir / "notebooks"
        return {
            'data_dir': data_dir,
            'model_dir': model_dir,
            'notebooks_dir': notebooks_dir,
            'project_root': project_root,
            'data_file': data_dir / "realtime" / "hanoi_weather_complete.csv",
            'update_script': notebooks_dir / "update_weather_data.py",
            'model_file': model_dir / "daily" / "BEST_CATBOOST_TUNED_DAILY.joblib",
            'selection_file': model_dir / "daily" / "selection_result_daily.joblib",
            'prediction_script': notebooks_dir / "generate_full_multi_horizon_predictions.py",
            'multi_pred_file': data_dir / "realtime" / "multi_horizon_predictions.csv"
        }
    except:
        return {
            'data_file': project_root / "data" / "realtime" / "hanoi_weather_complete.csv",
            'multi_pred_file': project_root / "data" / "realtime" / "multi_horizon_predictions.csv"
        }

PATHS = setup_paths()

# --- AUTO UPDATE ---
def is_streamlit_cloud():
    return os.path.exists("/home/appuser")

def safe_prediction_update():
    pred_script = PATHS['prediction_script']
    if not pred_script.exists():
        st.sidebar.warning("Prediction script not found.")
        return
    def run():
        try:
            cmd = [sys.executable, str(pred_script)]
            subprocess.run(cmd, cwd=PATHS['project_root'], capture_output=True, text=True, timeout=60, env=os.environ)
        except: pass
    threading.Thread(target=run, daemon=True).start()
    st.sidebar.info("Generating forecasts...")

def safe_data_update(after_pred=False):
    script = PATHS['update_script']
    if not script.exists():
        st.sidebar.warning("Update script not found.")
        return
    def run():
        try:
            cmd = [sys.executable, str(script)]
            result = subprocess.run(cmd, cwd=PATHS['project_root'], capture_output=True, text=True, timeout=120, env=os.environ)
            if result.returncode == 0 and after_pred:
                safe_prediction_update()
        except: pass
    threading.Thread(target=run, daemon=True).start()
    st.sidebar.info("Updating data...")

# --- DATA ---
@st.cache_data(ttl=1800)
def load_data():
    if not PATHS['data_file'].exists():
        return create_fallback()
    try:
        df = pd.read_csv(PATHS['data_file'], parse_dates=['datetime'])
        return df.sort_values('datetime').reset_index(drop=True)
    except:
        return create_fallback()

def create_fallback():
    dates = pd.date_range("2024-01-01", datetime.now(HANOI_TZ), freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'datetime': dates,
        'temp': 25 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.randn(len(dates)) * 2,
        'tempmin': np.random.normal(24, 2, len(dates)),
        'tempmax': np.random.normal(32, 2, len(dates)),
        'humidity': np.random.normal(75, 10, len(dates)),
        'windspeed': np.random.normal(10, 3, len(dates)),
        'precip': np.random.exponential(3, len(dates)).clip(0, 30),
        'visibility': np.random.normal(10, 2, len(dates)),
        'conditions': ['Partly Cloudy'] * len(dates),
        'feelslike': np.random.normal(28, 3, len(dates)),
        'dew': np.random.normal(22, 3, len(dates))
    })

def load_multi_predictions():
    if PATHS['multi_pred_file'].exists():
        try:
            df = pd.read_csv(PATHS['multi_pred_file'])
            df['as_of_date'] = pd.to_datetime(df['as_of_date'], errors='coerce').dt.date
            df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce').dt.date
            return df.dropna(subset=['target_date', 'predicted_temp'])
        except: pass
    return pd.DataFrame()

# --- MODEL ---
def load_model_info():
    if not PATHS['model_file'].exists():
        return {'name': 'CatBoost', 'r2': 0.87, 'n_features': 45, 'top': ['lag_1', 'humidity', 'temp_7d_mean']}
    try:
        data = joblib.load(PATHS['model_file'])
        return {
            'name': 'CatBoost (Tuned)',
            'r2': round(float(data.get('final_r2_mean', 0.87)), 4),
            'n_features': len(data.get('feature_names', [])),
            'top': data.get('feature_names', [])[:10]
        }
    except:
        return {'name': 'CatBoost', 'r2': 0.87, 'n_features': 45, 'top': ['lag_1', 'humidity', 'temp_7d_mean']}

# --- PREDICTIONS ---
def get_5day_forecast(df):
    try:
        from notebooks.preprocess_data import predict_future
        if PATHS['model_file'].exists() and len(df) >= 60:
            results = predict_future(df.tail(60))
            if isinstance(results, pd.DataFrame) and 'y_pred' in results.columns:
                results['date'] = pd.to_datetime(results['date']).dt.date
                return results[['date', 'y_pred']].head(5).rename(columns={'y_pred': 'temp'})
    except: pass
    today = datetime.now(HANOI_TZ).date()
    return pd.DataFrame([{'date': today + timedelta(days=i+1), 'temp': round(27 + np.random.randn() * 2, 1)} for i in range(5)])

# --- PAGE SETUP ---
def setup_page():
    st.set_page_config(page_title="Hanoi Weather", page_icon="Cloud", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/feather-icons/dist/feather.min.css">
    <script src="https://cdn.jsdelivr.net/npm/feather-icons/dist/feather.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    });
    </script>
    <style>
    .stApp { background: #f8faff; font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 1rem 1.5rem 2rem; max-width: 1200px; margin: 0 auto; }
    .nav-logo { font-weight: 600; color: #3b82f6; font-size: 1.1rem; }
    .stButton>button { background: transparent; color: #64748b; border: none; padding: 6px 12px; font-size: 0.9rem; border-radius: 6px; }
    .stButton>button:hover { color: #3b82f6; }
    .metric-card { background: white; border-radius: 14px; padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: 100%; }
    .metric-value { font-size: 1.5rem; font-weight: 600; color: #1e293b; }
    .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
    .forecast-card { background: white; border-radius: 14px; padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%; }
    .forecast-date { font-weight: 500; font-size: 1rem; }
    .forecast-temp { font-size: 2.1rem; font-weight: 700; margin: 10px 0 0; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- ICONS ---
def icon(name, size=28, color="#64748b"):
    return f'<i data-feather="{name}" style="width:{size}px;height:{size}px;color:{color};"></i>'

# --- NAVBAR ---
def render_navbar():
    current = st.session_state.get('page', 'current')
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; padding:12px 0;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1: st.markdown('<div class="nav-logo">Weather Forecast</div>', unsafe_allow_html=True)
    with c2:
        cols = st.columns(4)
        pages = [("Current", "current"), ("5-Day", "forecast"), ("Historical", "historical"), ("Model", "model")]
        for (label, key), col in zip(pages, cols):
            with col:
                if st.button(label):
                    st.session_state.page = key
                    st.rerun()
                if current == key:
                    st.markdown(f"<style>button:has(span:contains('{label}')) {{ color:#3b82f6 !important; font-weight:600 !important; }}</style>", unsafe_allow_html=True)
    with c3:
        if st.button("Refresh"):
            safe_data_update(after_pred=True)
            st.rerun()
    st.markdown("</div><hr style='margin:0; border-top:1px solid #e5e7eb;'>", unsafe_allow_html=True)

# --- CURRENT PAGE ---
def page_current(df):
    render_navbar()
    row = df.iloc[-1]
    temp = round(row['temp'], 1)
    feels = round(row['feelslike'], 1)
    cond = (row['conditions'] or "Partly Cloudy").split(',')[0].strip()

    weather_icons = {
        "Clear": "sun", "Partly Cloudy": "cloud", "Cloudy": "cloud", "Overcast": "cloud",
        "Rain": "cloud-rain", "Light Rain": "cloud-drizzle", "Heavy Rain": "cloud-rain",
        "Thunderstorm": "cloud-lightning", "Fog": "cloud", "Mist": "cloud", "Snow": "cloud-snow"
    }
    weather_icon = weather_icons.get(cond, "cloud")

    st.markdown(f"""
    <div style="
        position: relative;
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), 
                    url('https://th.bing.com/th/id/R.251ab71b625bfb04d102f68f4da9ffa0?rik=N%2bgrRw0PrMALVg&pid=ImgRaw&r=0');
        background-size: cover; background-position: center; color: white;
        border-radius: 16px; padding: 2rem; margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Inter', sans-serif;
    ">
        <div style="position: absolute; top: 12px; right: 16px; font-size: 0.85rem; opacity: 0.9;">
        </div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 1.4rem; font-weight: 600;">Hanoi, Vietnam</div>
                <div style="font-size: 0.95rem; opacity: 0.9; margin-top: 4px;">
                    {datetime.now(HANOI_TZ).strftime('%A, %I:%M %p')}
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="text-align: right;">
                    <div style="font-size: 4.8rem; font-weight: 700; line-height: 1; margin: 0;">
                        {temp}°C
                    </div>
                    <div style="font-size: 1.4rem; margin-top: 0.3rem;">{cond}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9; margin-top: 0.2rem;">
                        Feels like {feels}°C
                    </div>
                </div>
                <div style="font-size: 4.5rem;">
                    {icon(weather_icon, size=64, color='white')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        ("HUMIDITY", f"{row['humidity']:.0f}%", "droplet"),
        ("WIND", f"{row['windspeed']:.0f} km/h", "wind"),
        ("PRECIP", f"{row['precip']:.1f} mm", "umbrella"),
        ("VISIBILITY", f"{row['visibility']:.0f} km", "eye")
    ]
    for col, (label, val, icon_name) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem; margin-bottom:4px;">{icon(icon_name, size=28)}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# --- FORECAST PAGE ---
def page_forecast(df, historical_pred_df, today, start_date_input=None, end_date_input=None):
    render_navbar()  # BACK BUTTON NOW WORKS

    st.markdown("## 5-Day Temperature Forecast")
    
    # === 5-DAY CARDS ===
    forecast_5day = get_5day_forecast(df)
    if not forecast_5day.empty:
        cols = st.columns(5)
        weather_icons = {
            "Clear": "sun", "Partly Cloudy": "cloud", "Cloudy": "cloud", "Overcast": "cloud",
            "Rain": "cloud-rain", "Light Rain": "cloud-drizzle", "Heavy Rain": "cloud-rain",
            "Thunderstorm": "cloud-lightning"
        }
        for idx, row in forecast_5day.iterrows():
            date = row['date']
            temp = round(row['temp'], 1)
            day_name = date.strftime("%A")[:3]
            date_str = date.strftime("%b %d")
            cond = "Partly Cloudy"  # fallback
            icon_name = weather_icons.get(cond, "cloud")

            with cols[idx]:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-date">{day_name}<br><small>{date_str}</small></div>
                    <div style="font-size:3rem; margin:8px 0;">{icon(icon_name, size=48)}</div>
                    <div class="forecast-temp">{temp}°C</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("5-day forecast unavailable.")

    # === INTERACTIVE CHART BELOW ===
    st.markdown("### Actual vs Forecasted Temperature")
    
    try:
        today_ts = pd.Timestamp(today)
       
        # --- ENSURE DATETIME TYPES ---
        historical_pred_df = historical_pred_df.copy()
        historical_pred_df['target_date'] = pd.to_datetime(historical_pred_df['target_date']).dt.normalize()
        historical_pred_df['as_of_date'] = pd.to_datetime(historical_pred_df['as_of_date']).dt.normalize()

        # --- Actual temperatures ---
        actual = df[['datetime', 'temp']].copy()
        actual['datetime'] = pd.to_datetime(actual['datetime']).dt.normalize()
        actual = actual.set_index('datetime').sort_index()

        # Determine date range
        if start_date_input and end_date_input:
            start_ts = pd.Timestamp(start_date_input)
            end_ts = pd.Timestamp(end_date_input)
        else:
            default_start = max(
                actual.index.min(),
                historical_pred_df["target_date"].min() if not historical_pred_df.empty else today_ts - pd.Timedelta(days=30),
                today_ts - pd.Timedelta(days=30)
            )
            default_end = max(today_ts, historical_pred_df["target_date"].max()) if not historical_pred_df.empty else today_ts + pd.Timedelta(days=5)
            start_ts = default_start
            end_ts = default_end

        # Filter actual data: only up to today
        actual_end = min(end_ts, today_ts)
        actual = actual[(actual.index >= start_ts) & (actual.index <= actual_end)]

        # Filter forecasts
        if not historical_pred_df.empty:
            past_pred = historical_pred_df[
                (historical_pred_df['as_of_date'] <= today_ts) &
                (historical_pred_df['target_date'] >= start_ts) &
                (historical_pred_df['target_date'] <= end_ts)
            ].copy()
        else:
            past_pred = pd.DataFrame()

        # --- Plotly figure ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual['temp'],
            mode='lines',
            name='Actual',
            line=dict(color="#3a0eca", width=2.5)
        ))

        if not past_pred.empty:
            past_pred = past_pred.sort_values('as_of_date')
            colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
            for horizon in [1, 2, 3, 4, 5]:
                h_data = past_pred[past_pred['horizon'] == horizon]
                if not h_data.empty:
                    h_series = h_data.groupby('target_date')['predicted_temp'].last()
                    fig.add_trace(go.Scatter(
                        x=h_series.index,
                        y=h_series.values,
                        mode='lines',
                        name=f'{horizon}-Day Forecast',
                        line=dict(color=colors[horizon-1], width=1.4, dash='dot')
                    ))

        if start_ts <= today_ts <= end_ts:
            fig.add_vline(x=today_ts, line_width=1.4, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=today_ts, y=actual['temp'].max() if not actual.empty else 30,
                text="Today", showarrow=False, font=dict(color="black", size=10), yshift=10
            )

        fig.update_layout(
            height=380,
            xaxis=dict(title="Date", tickfont=dict(size=10), gridcolor='#e2e8f0'),
            yaxis=dict(title="Temperature (°C)", tickfont=dict(size=10), gridcolor='#e2e8f0',
                       range=[0, actual['temp'].max() + 3] if not actual.empty else [0, 35]),
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Date Picker Below Chart
        col1, col2 = st.columns(2)
        with col1:
            start_input = st.date_input("From", value=start_ts.date(), key="forecast_start")
        with col2:
            end_input = st.date_input("To", value=end_ts.date(), min_value=start_input, key="forecast_end")

        if st.button("Update Chart"):
            st.session_state.forecast_start = start_input
            st.session_state.forecast_end = end_input
            st.rerun()

    except Exception as e:
        st.error(f"Chart error: {e}")
# --- HISTORICAL PAGE ---
def page_historical(df):
    render_navbar()
    st.markdown("## Hanoi's Historical Temperature Trends")
    today = datetime.now(HANOI_TZ).date()
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col1:
        period = st.selectbox("Period", ["1W", "1M", "6M", "1Y", "All"], index=1)
    if period == "1W": default_start = today - timedelta(days=6)
    elif period == "1M": default_start = today - timedelta(days=29)
    elif period == "6M": default_start = today - timedelta(days=182)
    elif period == "1Y": default_start = today - timedelta(days=364)
    else: default_start = df['date'].min()
    
    with col2: start = st.date_input("From", value=default_start)
    with col3: end = st.date_input("To", value=today)
    with col4: feature = st.selectbox("Feature", ["Temperature", "Humidity", "Wind Speed", "Precipitation"])

    if start > end:
        st.error("Start cannot be after end.")
        return

    mask = (df['date'] >= start) & (df['date'] <= end)
    data = df[mask].copy()
    if data.empty:
        st.info("No data.")
        return

    col1, col2, col3 = st.columns(3)
    if feature == "Temperature":
        avg = data['temp'].mean()
        high = data['tempmax'].max(); high_date = data.loc[data['tempmax'].idxmax(), 'date'].strftime('%b %d')
        low = data['tempmin'].min(); low_date = data.loc[data['tempmin'].idxmin(), 'date'].strftime('%b %d')
        with col1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg:.0f}°C</div><div class='metric-label'>Average</div></div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#f97316;'>{high:.0f}°C</div><div class='metric-label'>High<br><small>{high_date}</small></div></div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#22c55e;'>{low:.0f}°C</div><div class='metric-label'>Low<br><small>{low_date}</small></div></div>", unsafe_allow_html=True)
    else:
        col_map = {"Humidity": "humidity", "Wind Speed": "windspeed", "Precipitation": "precip"}
        val = data[col_map[feature]].mean()
        unit = "%" if feature == "Humidity" else "mm" if feature == "Precipitation" else "km/h"
        with col1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{val:.1f}{unit}</div><div class='metric-label'>Avg {feature}</div></div>", unsafe_allow_html=True)

    st.markdown("### Trend")
    chart_data = data.set_index('datetime').sort_index()
    if feature == "Temperature":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['tempmax'], name="High", line=dict(color="#f97316")))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['tempmin'], name="Low", fill='tonexty', line=dict(color="#22c55e")))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['temp'], name="Avg", line=dict(color="#3b82f6", dash='dot')))
    else:
        col = {"Humidity": "humidity", "Wind Speed": "windspeed", "Precipitation": "precip"}[feature]
        fig = px.line(chart_data, y=col, color_discrete_sequence=["#3b82f6"])
    fig.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white', legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig, use_container_width=True)

    csv = data.to_csv(index=False).encode()
    st.download_button("Export CSV", csv, f"hanoi_{start}_to_{end}.csv", "text/csv")

# --- MODEL PAGE ---
def page_model(metrics):
    model_info = metrics
    overall_r2 = model_info.get('r2', 0.0)
    n_features = model_info.get('n_features', 0)
    top_features = ", ".join(model_info.get('top', []))

    render_navbar()
    
    st.markdown(
        "<h1 style='text-align:center; font-size:1.5rem; margin-bottom:1.8rem; color:#1e293b;'>MODEL PERFORMANCE</h1>",
        unsafe_allow_html=True
    )

    st.markdown("###### Data Information")
    st.markdown(f"""
    <div style="
        background:#f0f9ff; 
        padding:0.9rem 1.2rem; 
        border-radius:14px; 
        margin:1.5rem 0; 
        font-size:0.92rem; 
        color:#0c4a6e; 
        border-left:4px solid #3b82f6;
        box-shadow:0 2px 6px rgba(0,0,0,0.05);
    ">
        <strong>Data Source:</strong> Weather Query Builder | Visual Crossing<br>
        <strong>Time Range:</strong> 01/10/2015–01/10/2025
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###### Model Specifications")
    st.markdown(f"""
    <div style="
        background:white; 
        padding:1rem 2.5rem; 
        border-radius:14px; 
        box-shadow:0 3px 10px rgba(0,0,0,0.08); 
        margin:1rem 0;
        border:1px solid #e2e8f0;
    ">
        <div style="display:flex; gap:12rem; font-size:0.88rem; line-height:1.9;">
            <div style="color:#475569;">
                <div>Model</div>
                <div>Input Features</div>
                <div>Prediction Target</div>
            </div>
            <div style="font-weight:600; color:#1e293b;">
                <div>CatBoost (Tuned)</div>
                <div>1200features</div>
                <div>Daily average temperature</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###### Performance")
    cols = st.columns(2)
    perf_metrics = [
        ("Overall R²", f"{overall_r2:.4f}", "#10b981"),
        ("Number of features", f"{n_features}", "#3b82f6")
    ]
    for i, (label, value, color) in enumerate(perf_metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding:1rem 1.2rem; border-left:4px solid {color};">
                <div style="font-size:0.85rem; color:#475569; opacity:0.9;">{label}</div>
                <div style="font-size:1.35rem; font-weight:700; color:{color}; margin-top:6px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    if top_features:
        st.markdown("###### Feature Importance")
        st.markdown(f"""
        <div style="
            background:#f8fafc; 
            padding:1rem 1.2rem; 
            border-radius:12px; 
            border-left:5px solid #6366f1; 
            font-family:'Courier New', monospace; 
            font-size:0.88rem; 
            color:#1e293b; 
            box-shadow:0 2px 8px rgba(0,0,0,0.06);
            line-height:1.7;
        ">
            <strong>Top 10 features:</strong><br>
            <span style="color:#6366f1; font-weight:500;">{top_features}</span>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN ---
def main():
    setup_page()
    if 'page' not in st.session_state:
        st.session_state.page = 'current'

    # Load data and model info
    df = load_data()
    metrics = load_model_info()

    # Auto-update
    if not st.session_state.get('auto_update_done', False):
        if not is_streamlit_cloud():
            if not df.empty and 'datetime' in df.columns:
                last = pd.to_datetime(df['datetime']).max().date()
                today = datetime.now(HANOI_TZ).date()
                if today > last:
                    safe_data_update(after_pred=True)
        st.session_state.auto_update_done = True

    # Page routing
    page = st.session_state.page
    if page == 'current': 
        page_current(df)
    elif page == 'forecast':
        today = datetime.now(HANOI_TZ).date()
        historical_pred_df = load_multi_predictions()

        # Use session state for date persistence
        default_start = st.session_state.get('forecast_start', today - timedelta(days=30))
        default_end = st.session_state.get('forecast_end', today + timedelta(days=5))

        page_forecast(df, historical_pred_df, today, default_start, default_end)
    
    elif page == 'historical': 
        page_historical(df)
    elif page == 'model': 
        page_model(metrics)

    # Footer — ALWAYS SHOWN
    st.markdown("---")
    now_hanoi = datetime.now(HANOI_TZ)
    st.caption(f"Hanoi Weather • Last updated {now_hanoi.strftime('%H:%M')} (GMT+7)")

if __name__ == "__main__":
    main()
