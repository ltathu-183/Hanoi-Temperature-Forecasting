@echo off
echo Starting Hanoi Weather Forecast App...
cd /d "D:\dseb\5th_semester\ml1\Hanoi-Temperature-Forecasting"
call venv\Scripts\activate.bat
streamlit run ui\scripts\app.py
pause