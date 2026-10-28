@echo off
title AgriSense Agent - Launch Web Application
cd /d "%~dp0"

echo ========================================================
echo         🌾 AgriSense Agent - Web Application
echo ========================================================
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Starting Streamlit server...
streamlit run app.py
pause
