"""
Crop Yield Prediction Engine for AgriSense Agent.
Wraps the trained scikit-learn RandomForestRegressor pipeline, formats predictions
with explicit units (hg/ha and tonnes/ha), and appends agronomic limitation notes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import pandas as pd
from pydantic import BaseModel, Field


class CropYieldInput(BaseModel):
    area: str = Field(..., description="Country or Region name, e.g. India, United States, France")
    crop: str = Field(..., description="Crop name, e.g. Maize, Potatoes, Wheat, Rice, paddy")
    year: int = Field(default=2024, ge=1960, le=2050, description="Target crop harvest year")
    rainfall_mm: float = Field(..., ge=0.0, le=8000.0, description="Average annual rainfall in mm")
    pesticides_tonnes: float = Field(..., ge=0.0, description="Estimated pesticides used in tonnes")
    temperature_celsius: float = Field(..., ge=-10.0, le=55.0, description="Average temperature in Celsius")


class CropYieldOutput(BaseModel):
    area: str
    crop: str
    year: int
    yield_hg_per_ha: float
    yield_tonnes_per_ha: float
    unit_labels: Dict[str, str]
    confidence_tier: str
    agronomic_notes: List[str]
    input_parameters: Dict[str, Any]


class YieldPredictor:
    """Manages model loading and prediction operations."""

    def __init__(self, model_path: str = "artifacts/yield_model.joblib", metrics_path: str = "artifacts/model_metrics.json"):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.bundle: Optional[Dict[str, Any]] = None
        self.pipeline = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.load_model()

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def load_model(self) -> bool:
        if not self.model_path.exists():
            try:
                from src.train_model import train_yield_model
                train_yield_model(output_dir=str(self.model_path.parent))
            except Exception as e:
                print(f"Notice: Auto-train on start notice: {e}")

        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, dict) and "pipeline" in loaded:
                    self.bundle = loaded
                    self.pipeline = loaded["pipeline"]
                else:
                    self.pipeline = loaded
                    self.bundle = {}
                if self.metrics_path.exists():
                    with open(self.metrics_path, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                return True
            except Exception as e:
                print(f"Warning: Failed to load model bundle: {e}")
                return False
        return False


    def get_supported_areas(self) -> List[str]:
        if self.bundle and "supported_areas" in self.bundle:
            return self.bundle["supported_areas"]
        if self.metrics and "supported_areas" in self.metrics:
            return self.metrics["supported_areas"]
        return [
            "Albania", "Algeria", "Angola", "Argentina", "Australia", "Brazil",
            "Canada", "Egypt", "France", "India", "Indonesia", "Japan",
            "Kenya", "Mexico", "Nigeria", "Pakistan", "South Africa", "Spain",
            "United Kingdom", "United States"
        ]

    def get_supported_crops(self) -> List[str]:
        if self.bundle and "supported_crops" in self.bundle:
            return self.bundle["supported_crops"]
        if self.metrics and "supported_crops" in self.metrics:
            return self.metrics["supported_crops"]
        return [
            "Cassava", "Maize", "Potatoes", "Rice, paddy", "Sorghum",
            "Soybeans", "Sweet potatoes", "Wheat", "Yams"
        ]

    def predict(self, req: CropYieldInput) -> CropYieldOutput:
        """
        Generates a crop yield prediction given agronomic and weather inputs.
        Falls back to calibrated baselines if model file is pending training.
        """
        notes: List[str] = []

        # Construct dataframe row matching the exact pipeline feature schema
        row = pd.DataFrame([
            {
                "Area": req.area,
                "Item": req.crop,
                "Year": req.year,
                "average_rain_fall_mm_per_year": req.rainfall_mm,
                "pesticides_tonnes": req.pesticides_tonnes,
                "avg_temp": req.temperature_celsius,
            }
        ])

        if self.pipeline is not None:
            raw_prediction = float(self.pipeline.predict(row)[0])
            # Ensure non-negative physical yield
            yield_hg = max(0.0, raw_prediction)
            confidence = "High (Supervised ML Ensemble)"
        else:
            # Calibrated baseline estimation (hg/ha) if pipeline is not yet fitted
            baseline_map = {
                "Potatoes": 240000.0,
                "Sweet potatoes": 120000.0,
                "Cassava": 110000.0,
                "Yams": 105000.0,
                "Rice, paddy": 45000.0,
                "Maize": 38000.0,
                "Wheat": 28000.0,
                "Soybeans": 22000.0,
                "Sorghum": 18000.0,
            }
            base = baseline_map.get(req.crop, 30000.0)
            # Weather adjustment heuristics
            temp_mult = 1.0 - (abs(req.temperature_celsius - 22.0) * 0.015)
            rain_mult = 1.0 if 500 <= req.rainfall_mm <= 1600 else 0.82
            yield_hg = max(1000.0, base * max(0.4, temp_mult) * rain_mult)
            confidence = "Heuristic Baseline (Model Training Pending)"
            notes.append("Trained joblib pipeline was not found; prediction uses FAO historical agronomic baseline.")

        # Unit conversion: 10,000 hectograms/ha = 1 metric tonne/ha
        yield_tonnes = round(yield_hg / 10000.0, 3)
        yield_hg_rounded = round(yield_hg, 1)

        # Agronomic domain limitation notes
        if req.rainfall_mm < 350:
            notes.append("Low rainfall detected (<350 mm/yr): Rainfed crops may suffer severe water stress; supplemental irrigation strongly advised.")
        elif req.rainfall_mm > 2400:
            notes.append("Excess rainfall detected (>2400 mm/yr): Monitor for waterlogging, nitrogen leaching, and root rot.")

        if req.temperature_celsius > 35.0:
            notes.append("Extreme heat condition (>35°C): Heat stress during pollination stage can sharply reduce actual grain fill.")
        elif req.temperature_celsius < 8.0 and req.crop not in ["Wheat", "Potatoes"]:
            notes.append("Chilling threshold: Warm-season crops (Maize/Rice/Soybean) have retarded growth below 10°C.")

        if req.pesticides_tonnes < 100 and req.area in ["United States", "Brazil", "India"]:
            notes.append("Low pesticide quota relative to national average: Ensure Integrated Pest Management (IPM) to mitigate biotic losses.")

        return CropYieldOutput(
            area=req.area,
            crop=req.crop,
            year=req.year,
            yield_hg_per_ha=yield_hg_rounded,
            yield_tonnes_per_ha=yield_tonnes,
            unit_labels={
                "hg_per_ha": "hectograms per hectare (hg/ha)",
                "tonnes_per_ha": "metric tonnes per hectare (t/ha)"
            },
            confidence_tier=confidence,
            agronomic_notes=notes,
            input_parameters={
                "Area": req.area,
                "Crop": req.crop,
                "Year": req.year,
                "Annual Rainfall (mm)": req.rainfall_mm,
                "Pesticide Usage (tonnes)": req.pesticides_tonnes,
                "Average Temperature (°C)": req.temperature_celsius
            }
        )
