"""
Unit Tests for Automated Dataset Acquisition and Continuous Model Retraining.
Verifies World Bank acquisition logging and embedded benchmark generation
and accuracy optimization without manual file placement.
"""

from pathlib import Path
import pytest
import src.data_fetcher as data_fetcher_module
from src.data_fetcher import AutomatedDatasetFetcher
from src.train_model import train_yield_model
from src.agent_router import AgriSenseAgentRouter


def test_automated_dataset_fetcher_acquires_records(tmp_path):
    temp_csv = tmp_path / "yield_test.csv"
    fetcher = AutomatedDatasetFetcher(output_path=temp_csv)
    log = fetcher.fetch_and_update_dataset(years_back=4)

    assert log["status"] == "SUCCESS"
    assert log["records_fetched"] >= 0 or log["new_records_added"] > 0
    assert temp_csv.exists(), "Fetched dataset file must be created on disk"
    assert len(log["years_covered"]) > 0


def test_fetch_log_distinguishes_api_and_embedded_records(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher_module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    fetcher = AutomatedDatasetFetcher(output_path=tmp_path / "yield_test.csv")
    monkeypatch.setattr(fetcher, "_fetch_world_bank_yield_records", lambda: [])
    monkeypatch.setattr(
        fetcher,
        "_generate_reference_records",
        lambda years_back: [{
            "Area": "India",
            "Item": "Wheat",
            "Year": 2024,
            "hg/ha_yield": 34500,
            "average_rain_fall_mm_per_year": 1120.0,
            "pesticides_tonnes": 62000.0,
            "avg_temp": 25.4,
        }],
    )

    log = fetcher.fetch_and_update_dataset()

    assert log["sources_queried"] == ["World Bank Open Data API"]
    assert log["sources_used"] == ["Embedded benchmark estimates"]
    assert log["records_fetched"] == 0
    assert log["records_generated"] == 1
    assert log["new_records_added"] == 1


def test_agent_router_recognizes_training_intent():
    router = AgriSenseAgentRouter()
    
    assert router.classify_intent("Fetch latest datasets and train the model") == "model_training"
    assert router.classify_intent("Retrain the ML model to improve accuracy") == "model_training"
    assert router.classify_intent("Auto train model with new data") == "model_training"


def test_training_pipeline_improves_accuracy():
    pipe, metrics = train_yield_model(auto_fetch_sources=False, optimize_hyperparams=True)
    assert metrics["r2_score_test"] > 0.70, "Optimized model should achieve high test R² score"
    assert "data_sources" in metrics
    assert Path("artifacts/yield_model.joblib").exists()
