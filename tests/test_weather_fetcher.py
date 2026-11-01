from src.weather_fetcher import CLIMATE_MODELS, WeatherFetchError, WeatherFetcher


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_annual_weather_uses_location_and_median_of_climate_models(monkeypatch):
    day_count = 365
    climate_data = {
        "time": ["2025-01-01"] * day_count,
    }
    for model, daily_rain, daily_temperature in zip(
        CLIMATE_MODELS, (1.0, 2.0, 3.0), (20.0, 22.0, 24.0)
    ):
        climate_data[f"precipitation_sum_{model}"] = [daily_rain] * day_count
        climate_data[f"temperature_2m_mean_{model}"] = [daily_temperature] * day_count

    responses = iter(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "name": "Coimbatore",
                            "country": "India",
                            "latitude": 11.0,
                            "longitude": 77.0,
                        }
                    ]
                }
            ),
            FakeResponse({"daily": climate_data}),
        ]
    )
    request_calls = []

    def fake_get(url, params, timeout):
        request_calls.append((url, params, timeout))
        return next(responses)

    monkeypatch.setattr("src.weather_fetcher.requests.get", fake_get)

    result = WeatherFetcher().fetch_annual_weather("India", 2025, "Coimbatore")

    assert result.rainfall_mm == 730.0
    assert result.temperature_celsius == 22.0
    assert result.resolved_location == "Coimbatore"
    assert request_calls[0][1]["name"] == "Coimbatore, India"
    assert request_calls[1][1]["start_date"] == "2025-01-01"
    assert request_calls[1][1]["end_date"] == "2025-12-31"


def test_annual_weather_rejects_incomplete_climate_data(monkeypatch):
    responses = iter(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "name": "India",
                            "country": "India",
                            "latitude": 22.0,
                            "longitude": 79.0,
                        }
                    ]
                }
            ),
            FakeResponse({"daily": {"time": ["2025-01-01"]}}),
        ]
    )
    monkeypatch.setattr(
        "src.weather_fetcher.requests.get",
        lambda url, params, timeout: next(responses),
    )

    try:
        WeatherFetcher().fetch_annual_weather("India", 2025)
    except WeatherFetchError as exc:
        assert "incomplete daily data" in str(exc)
    else:
        raise AssertionError("Incomplete climate data must be reported.")


def test_annual_weather_rejects_year_outside_api_range():
    try:
        WeatherFetcher().fetch_annual_weather("India", 2050)
    except ValueError as exc:
        assert "1950 through 2049" in str(exc)
    else:
        raise AssertionError("Years outside the climate API range must be rejected.")
