# AgriSense Agent - PowerShell Setup Script
Write-Host "========================================================" -ForegroundColor Green
Write-Host "       🌾 AgriSense Agent - Initialization Script" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

# 1. Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[1/5] Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "[1/5] Virtual environment already exists." -ForegroundColor Gray
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# 2. Dependencies
Write-Host "`n[2/5] Installing project dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# 3. Data Audit
Write-Host "`n[3/5] Running Data Audit on Kaggle dataset..." -ForegroundColor Cyan
python src/data_audit.py

# 4. Model Training
Write-Host "`n[4/5] Training RandomForestRegressor Pipeline..." -ForegroundColor Cyan
python src/train_model.py

# 5. Tests
Write-Host "`n[5/5] Executing Pytest Acceptance Gates..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "  ✅ AgriSense Initialized Successfully!" -ForegroundColor Green
Write-Host "  Start the UI with: streamlit run app.py" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Green
