# 🌾 AgriSense Agent
> **Autonomous Agricultural Intelligence & Crop Yield Prediction System**  
> *Minimal-Resource Autonomous Agent Blueprint for Google Antigravity & Local Workstations.*

---

## 📌 Executive Summary & Architecture

**AgriSense** combines a structured machine learning **crop-yield predictor** with a local **Retrieval-Augmented Generation (RAG)** knowledge base and a **live web search fallback**:

1. **Numeric Crop Yield Prediction (Primary Focus):**
   - Trained on the project’s local crop-yield dataset (`Area`, `Item`, `Year`, rainfall, pesticides, temperature, and `hg/ha_yield`).
   - Supervised ML Pipeline with `scikit-learn` (`ColumnTransformer` + `RandomForestRegressor`).
   - Outputs explicit dual units: **tonnes/hectare (t/ha)** and **hectograms/hectare (hg/ha)** alongside agronomic limitation notes (e.g. drought stress, temperature caps).
   - Loads annual rainfall and mean-temperature estimates automatically for the selected year and location; manual weather overrides are optional.

2. **Year- and Location-Based Climate Inputs:**
   - Uses Open-Meteo geocoding plus its Climate API to retrieve daily climate-model data for the selected country and optional city/town.
   - Aggregates the daily data into annual rainfall and mean temperature using the median across three climate models.
   - Climate-model values are scenario estimates, not day-specific weather forecasts. The crop-yield model is trained on historical data, so predictions beyond its latest training year are clearly marked as unvalidated future scenarios.

3. **Automated Dataset Ingestion and Inspection:**
   - Fetches cereal-yield rows from the **World Bank Open Data API** when available and combines them with embedded crop benchmark estimates in [`src/data_fetcher.py`](./src/data_fetcher.py). The embedded estimates are not live FAOSTAT downloads.
   - Saves the combined training dataset to `data/raw/yield_df.csv`; the **Model & Data Diagnostics** tab displays its rows and columns and lets you download the CSV.
   - Retrains the `RandomForestRegressor` with optional hyperparameter search using that persisted dataset.

4. **Crop Yield Knowledge Base (Local RAG):**
   - Curated authoritative documents in `knowledge_docs/` focused strictly on **Crop Yield Prediction, Yield Modeling, Benchmarks, Rainfall/Temperature Impact Curves, and Yield Gaps**.
   - Offline, fast semantic search with source titles, publishers, dates, and active links.

5. **Live Search for Other Agricultural Aspects:**
   - If an inquiry is outside crop yield prediction (e.g., pests, crop diseases, fertilizer brands, market prices, irrigation technology), the agent dynamically searches live online sources.
   - Powered by **Tavily Web Search** or **Free DuckDuckGo Search (100% free, 0 keys needed)**.

6. **Dedicated API Key Center:**
   - Kept in a dedicated place (`.env` and in the interactive **API Keys & Settings** tab in the Streamlit UI).
   - Compatible with **100% Free Tiers** (Tavily free tier, Google Gemini free AI Studio, Groq free tier).
   - Zero cost required to run the full application.

7. **Agriculture-Only Guardrails:**
   - Deterministic scope gate refusing out-of-scope queries (general coding, medical advice, entertainment).
   - Ingestion sanitization to mitigate prompt injection.


---

## 📁 Project Directory Structure

```text
Agri Sense Agent/
├── .env                          # Local secrets and API keys (never committed)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules for secrets and heavy binaries
├── requirements.txt              # All project dependencies
├── setup.bat                     # Windows 1-click environment setup & test runner
├── setup.ps1                     # PowerShell setup script
├── run_app.bat                   # 1-click Streamlit web launcher
├── app.py                        # Streamlit UI (Prediction Form + Chat + API Key Center)
├── README.md                     # Project documentation
│
├── data/
│   └── raw/
│       └── yield_df.csv          # Persisted training dataset (World Bank + embedded estimates)
│
├── artifacts/
│   ├── audit_report.json         # Data audit summary and column stats
│   ├── model_metrics.json        # Test R² score, MAE, and RMSE metrics
│   └── yield_model.joblib        # Trained scikit-learn Pipeline bundle
│
├── knowledge_docs/               # Curated Crop Yield Prediction RAG Documents
│   ├── 01_crop_yield_prediction_methodologies.txt
│   ├── 02_cereal_yield_benchmarks_maize_wheat_rice.txt
│   ├── 03_tuber_yield_benchmarks_potatoes_cassava.txt
│   ├── 04_rainfall_impact_on_crop_yields.txt
│   ├── 05_temperature_and_thermal_time_yield_effects.txt
│   ├── 06_pesticide_efficiency_and_yield_protection.txt
│   ├── 07_yield_units_and_conversions.txt
│   ├── 08_regional_yield_gap_analysis.txt
│   ├── 09_soybean_and_legume_yield_modeling.txt
│   ├── 10_sorghum_and_coarse_cereal_yield_prediction.txt
│   └── 11_soil_organic_matter_and_yield_plateaus.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized API key and directory management
│   ├── data_fetcher.py           # World Bank retrieval and embedded benchmark generation
│   ├── weather_fetcher.py        # Location/year climate estimates from Open-Meteo
│   ├── data_audit.py             # Schema, missing values, and column verification
│   ├── train_model.py            # RandomForestRegressor training & hyperparameter search
│   ├── yield_predictor.py        # Inference engine with units & limitation notes
│   ├── search_service.py         # Free live online agricultural search
│   ├── rag_engine.py             # RAG knowledge base & search router
│   ├── guardrails.py             # Agriculture-only scope gate & prompt security
│   └── agent_router.py           # Master query coordinator & intent classifier
│
└── tests/
    ├── __init__.py
    ├── test_data_fetcher.py      # Verifies automated data acquisition & continuous retraining
    ├── test_yield_predictor.py   # Verifies finite numbers, units, and limitations
    ├── test_guardrails.py        # Verifies scope refusals and injection safety
    ├── test_rag.py               # Verifies clickable URLs and search fallbacks
    └── test_security.py          # Verifies no API keys are exposed

```

---

## ⚡ Quick Start Guide

### Step 1: Initialize Environment & Train Model
Run the setup script:
```powershell
.\setup.bat
```
*(Or in PowerShell: `.\setup.ps1`)*

This script automatically:
1. Creates the Python virtual environment (`venv`).
2. Installs dependencies from `requirements.txt`.
3. Runs the data audit on `data/raw/yield_df.csv`.
4. Trains the `scikit-learn` `RandomForestRegressor` model pipeline and logs metrics.
5. Executes the `pytest` suite against all acceptance criteria.

---

### Step 2: Launch the Web Application
Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔑 Free API Keys & Configuration

All API keys can be managed either in `.env` or interactively inside the **"🔑 API Keys & Free Services"** tab in the Streamlit application:

| Service | Free Tier Allowance | How to Obtain | Purpose in AgriSense |
|---|---|---|---|
| **DuckDuckGo Search** | **100% Free / Unlimited** | *No key or signup needed* | Default live search for general agricultural topics |
| **Tavily Web Search** | 1,000 queries/month free | [tavily.com](https://tavily.com) | Live agronomic web search fallback |
| **Google Gemini** | 15 RPM free | [aistudio.google.com](https://aistudio.google.com) | Free cloud LLM synthesis |
| **Groq Cloud** | Free fast tier | [console.groq.com](https://console.groq.com) | Ultra-low latency reasoning |

> 💡 **Notice:** You can enter your API keys whenever you are ready. The application will function immediately with its offline RAG and DuckDuckGo free search fallbacks.

---

## 🧪 Testing Acceptance Criteria

Execute the automated test suite anytime with:
```bash
pytest tests/ -v
```

1. **Gate 1 - Finite Predictions with Units**: `tests/test_yield_predictor.py` validates finite non-negative numbers in both `tonnes/ha` and `hg/ha`.
2. **Gate 2 - Clickable Source URLs**: `tests/test_rag.py` ensures all retrieved answers provide valid `http://` or `https://` citation links.
3. **Gate 3 - Clean Scope Refusals**: `tests/test_guardrails.py` tests automatic rejection of coding, medicine, and prompt injections.
4. **Gate 4 - Zero Leaked Keys**: `tests/test_security.py` confirms `.gitignore` covers secrets and scans source files for keys.
