# generate_full_multi_horizon_predictions.py
"""
Generate historical multi-horizon predictions.
- If output exists, only append new predictions (incremental)
- If output exists, default start date is 2 October 2025
- Only run if data is newer than last prediction
"""

import pandas as pd
import sys
from pathlib import Path
import yaml
from datetime import datetime, date
from zoneinfo import ZoneInfo  # Python 3.9+, built-in and works perfectly on Streamlit Cloud
# or fallback for very old Python: import pytz; HANOI_TZ = pytz.timezone('Asia/Ho_Chi_Minh')
HANOI_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

def load_config() -> dict:
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    try:
        CONFIG = load_config()
        print(" Configuration loaded")
    except Exception as e:
        print(f" Config error: {e}")
        return

    # --- Handle paths ---
    required_paths = ["realtime_data", "model_file"]
    for p in required_paths:
        if p not in CONFIG.get("paths", {}):
            print(f" '{p}' missing in config['paths']")
            return

    data_file = project_root / CONFIG["paths"]["realtime_data"]
    model_file = project_root / CONFIG["paths"]["model_file"]
    
    output_file = project_root / CONFIG.get(
        "paths", {}
    ).get("historical_predictions", "data/realtime/multi_horizon_predictions.csv")

    # Validate files
    if not data_file.exists():
        print(f" Data file not found: {data_file}")
        return
    if not model_file.exists():
        print(f" Model file missing: {model_file}")

    # Load data
    try:
        df = pd.read_csv(data_file, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        last_data_date = df['datetime'].max().date()
        today = date.today(HANOI_TZ)
        print(f" Data last updated: {last_data_date} (today: {today})")
    except Exception as e:
        print(f" Data loading failed: {e}")
        return

    # Import predictor
    try:
        from notebooks.preprocess_data import predict_future
    except Exception as e:
        print(f" Import failed: {e}")
        return

    min_history = CONFIG.get("model", {}).get("required_history_days", 365)
    if len(df) < min_history + 5:
        print(f" Not enough data. Need {min_history + 5} days, have {len(df)}")
        return

    default_start_date = date(2025, 10, 2)
    existing_df = pd.DataFrame()

    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, parse_dates=["as_of_date", "target_date"])
            print(" Existing predictions file found.")

            if not existing_df.empty:
                last_as_of = existing_df["as_of_date"].max().date()
                effective_start_date = last_as_of + pd.Timedelta(days=1)
                print(f" Existing predictions up to {last_as_of}.")
            else:
                # file exists but empty → start at default
                effective_start_date = default_start_date
                print(" Predictions file empty → starting at default date.")
        except Exception as e:
            print(f" Error loading predictions. Starting at default. {e}")
            effective_start_date = default_start_date
    else:
        # No file → start at default
        print(" No predictions file → starting at default date.")
        effective_start_date = default_start_date

    # Convert start date to index
    # Find first index where datetime >= effective_start_date
    valid_indices = df[df["datetime"].dt.date >= effective_start_date].index
    if valid_indices.empty:
        print(" No new dates to forecast.")
        return
    start_idx = valid_indices[0]
    print(f" Will start generating from {effective_start_date} at index {start_idx}")
    # -----------------------------------------------------------

    total_days = len(df)
    if start_idx >= total_days:
        print(" No new dates to forecast.")
        return

    # Generate new predictions
    print(f" Generating forecasts from index {start_idx} to {total_days}...")
    all_predictions = []

    for i in range(start_idx, total_days):
        try:
            history = df.iloc[:i+1]
            pred_result = predict_future(history)
            
            if isinstance(pred_result, pd.DataFrame) and len(pred_result) >= 5:
                as_of_date = df.iloc[i]["datetime"].date()
                for h in range(5):
                    raw_date = pred_result.iloc[h]["date"]
                    if isinstance(raw_date, str):
                        target_date = pd.to_datetime(raw_date).date()
                    else:
                        target_date = pd.to_datetime(raw_date).date() 
                    predicted_temp = float(pred_result.iloc[h]["y_pred"])
                    # Compute model_day = target_date - as_of_date
                    model_day = (target_date - as_of_date).days

                    all_predictions.append({
                        "as_of_date": as_of_date,
                        "target_date": target_date,
                        "horizon": h + 1,
                        "model_day": model_day,
                        "predicted_temp": predicted_temp
                    })

        except Exception as e:
            print(f"   Skip day {i}: {e}")
            continue

    # Save predictions
    if all_predictions:
        new_df = pd.DataFrame(all_predictions)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not existing_df.empty:
            full_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            full_df = new_df
        full_df = full_df.drop_duplicates(
            subset=["as_of_date", "target_date", "horizon"],
            keep="last"
        ).reset_index(drop=True)
        full_df.to_csv(output_file, index=False)
        print(f"\n Success! Added {len(new_df)} new predictions to:\n{output_file}")
    else:
        print("\n No new predictions generated.")

if __name__ == "__main__":
    main()
