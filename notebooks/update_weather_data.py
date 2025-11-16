import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import tempfile
import shutil
from io import StringIO  
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()  # This loads from current working directory by default

API_KEY = os.getenv("WEATHER_API_KEY")
print(f"API_KEY loaded: {'Yes' if API_KEY else 'No'}")
if API_KEY:
    print(f"API_KEY length: {len(API_KEY)}")
    print(f"API_KEY preview: {API_KEY[:5]}...{API_KEY[-5:] if len(API_KEY) > 10 else ''}")
else:
    print("API_KEY is None — check .env location!")

# -------------------------- CONFIG --------------------------
# REMOVED: Hardcoded API_KEY - using get_api_key() function instead
CITY = "Hanoi"
CHUNK_DAYS = 365
RETRY_ATTEMPTS = 3
SLEEP_BETWEEN_CHUNKS = 1

BASE_DIR = Path(__file__).parent.parent
EXISTING_DATA_PATH = BASE_DIR / "data" / "realtime" / "hanoi_weather_complete.csv"
MODEL_DATA = BASE_DIR / "data" / "daily" / "daily_data.csv"

def load_existing() -> pd.DataFrame:
    if EXISTING_DATA_PATH.exists():
        df = pd.read_csv(EXISTING_DATA_PATH, parse_dates=["datetime"])
    else:
        print(f"{EXISTING_DATA_PATH} not found. Loading from model data: {MODEL_DATA}")
        df = pd.read_csv(MODEL_DATA, parse_dates=["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} rows – last date: {df['datetime'].max().date()}")
    return df.copy()


def build_date_chunks(start: datetime, end: datetime) -> List[tuple]:
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks
def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{CITY}/{start}/{end}?"
        f"unitGroup=metric&include=days&key={API_KEY}&contentType=csv"  # ← CHANGED TO CSV
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                # Parse CSV directly from text response
                df = pd.read_csv(StringIO(r.text), low_memory=False)
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df.copy()
            else:
                print(f"Attempt {attempt+1}/{RETRY_ATTEMPTS} failed ({r.status_code}) – retrying...")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Network error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {start} to {end}")


def safe_save_csv(df: pd.DataFrame, path: Path, max_retries: int = 5) -> None:
    temp_file = path.with_suffix(".tmp")
    for attempt in range(max_retries):
        try:
            df_copy = df.copy()
            df_copy.to_csv(temp_file, index=False)
            shutil.move(temp_file, path)
            print(f"Saved to {path}")
            return
        except PermissionError:
            print(f"File in use. Retrying in 2s... ({attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"Save error: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to save {path}")


def main() -> None:
    df_raw = load_existing()
    last_raw_date = df_raw["datetime"].max().date()
    today = datetime.now().date()  # ← CHANGED: get TODAY

    if last_raw_date >= today:
        print("Dataset already up-to-date – nothing to do.")
        return

    start_date = last_raw_date + timedelta(days=1)
    print(f"Need data from {start_date} to {today}")  # ← includes today

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(today, datetime.min.time())  # ← up to today

    chunks = build_date_chunks(start_dt, end_dt)

    all_new_data = []

    for i, (s, e) in enumerate(chunks, 1):
        print(f"Chunk {i}/{len(chunks)}: {s} - {e}")
        chunk_df = fetch_chunk(s, e)
        print(f" {len(chunk_df)} days received")
        all_new_data.append(chunk_df)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    if all_new_data:
        new_df = pd.concat(all_new_data, ignore_index=True, copy=False)
    else:
        new_df = pd.DataFrame()
        
    print(f"Total new days fetched: {len(new_df)}")

    # Align columns
    if not new_df.empty:
        common_cols = df_raw.columns.intersection(new_df.columns)
        
        # Create new DataFrames with aligned columns (more efficient)
        aligned_raw = df_raw[common_cols].copy()
        aligned_new = new_df[common_cols].copy()
        
        # FIXED: Single concat operation for final merge
        full = pd.concat([aligned_raw, aligned_new], ignore_index=True, copy=False)
    else:
        full = df_raw.copy()

    full = (
        full
        .drop_duplicates(subset="datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
        .copy()  
    )

    safe_save_csv(full, EXISTING_DATA_PATH)

    print("\nSUCCESS!")
    print(f"   Total rows : {len(full)}")
    print(f"   Date range : {full['datetime'].min().date()} to {full['datetime'].max().date()}")
    print(f"   Updated    : {EXISTING_DATA_PATH}")

if __name__ == "__main__":
    main()