"""
Unit and Integration Tests for AgriSense Yield Predictor.
Verifies Acceptance Gate (1): Valid predictions output finite numbers with units.
"""

import math
import pytest
from src.yield_predictor import YieldPredictor, CropYieldInput


def test_yield_predictor_outputs_finite_numbers():
    predictor = YieldPredictor()
    req = CropYieldInput(
        area="India",
        crop="Maize",
        year=2024,
        rainfall_mm=1083.0,
        pesticides_tonnes=75000.0,
        temperature_celsius=25.0
    )
    result = predictor.predict(req)

    # 1. Output values must be finite positive numbers
    assert math.isfinite(result.yield_hg_per_ha), "Yield in hg/ha must be finite"
    assert math.isfinite(result.yield_tonnes_per_ha), "Yield in tonnes/ha must be finite"
    assert result.yield_hg_per_ha > 0, "Yield in hg/ha must be strictly positive"
    assert result.yield_tonnes_per_ha > 0, "Yield in tonnes/ha must be strictly positive"

    # 2. Conversion check: 10,000 hg/ha = 1 tonne/ha
    expected_tonnes = round(result.yield_hg_per_ha / 10000.0, 3)
    assert abs(result.yield_tonnes_per_ha - expected_tonnes) < 0.01, "Unit conversion must match 1 t/ha = 10,000 hg/ha"

    # 3. Unit labels must be explicitly populated
    assert "hg_per_ha" in result.unit_labels
    assert "tonnes_per_ha" in result.unit_labels


def test_yield_predictor_limitation_notes():
    predictor = YieldPredictor()
    # Test low rainfall trigger
    low_rain_req = CropYieldInput(
        area="Algeria",
        crop="Wheat",
        year=2024,
        rainfall_mm=150.0,
        pesticides_tonnes=1800.0,
        temperature_celsius=24.0
    )
    result = predictor.predict(low_rain_req)
    assert any("Low rainfall detected" in note for note in result.agronomic_notes), "Should flag drought/low rainfall limitation"


def test_supported_lists_exist():
    predictor = YieldPredictor()
    areas = predictor.get_supported_areas()
    crops = predictor.get_supported_crops()

    assert len(areas) >= 10, "Should support multiple countries/regions"
    assert len(crops) >= 5, "Should support core staple crops"
    assert "India" in areas or "United States" in areas
    assert "Maize" in crops and "Wheat" in crops


def test_future_year_is_reported_as_an_unvalidated_scenario():
    class ConstantPipeline:
        def predict(self, row):
            return [28000.0]

    predictor = YieldPredictor.__new__(YieldPredictor)
    predictor.pipeline = ConstantPipeline()
    predictor.metrics = {"years_represented": [2018, 2020, 2024]}

    result = predictor.predict(
        CropYieldInput(
            area="India",
            crop="Maize",
            year=2028,
            rainfall_mm=1100.0,
            pesticides_tonnes=75000.0,
            temperature_celsius=25.0,
        )
    )

    assert "Indicative Future Scenario" in result.confidence_tier
    assert any("beyond the latest training year (2024)" in note for note in result.agronomic_notes)
