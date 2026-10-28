@echo off
title AgriSense Agent - Setup & Pipeline Initialization
cd /d "%~dp0"

echo ========================================================
echo        🌾 AgriSense Agent - Initialization Script
echo ========================================================
echo.

if not exist venv (
    echo [1/5] Creating Python virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment already exists.
)

call venv\Scripts\activate.bat

echo.
echo [2/5] Installing project dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [3/5] Running Data Audit on data/raw/yield_df.csv...
python src/data_audit.py

echo.
echo [4/5] Training Scikit-Learn RandomForestRegressor Pipeline...
python src/train_model.py

echo.
echo [5/5] Running Pytest Verification Suite...
pytest tests/ -v

echo.
echo ========================================================
echo   ✅ Setup and Verification Completed Successfully!
echo   To launch the Streamlit app, run: run_app.bat
echo   Or type: streamlit run app.py
echo ========================================================
pause
