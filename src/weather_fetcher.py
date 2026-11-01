"""Location-based annual climate estimates for crop-yield scenarios."""

import calendar
import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
CLIMATE_MODELS = ("EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S")


class WeatherFetchError(RuntimeError):
    """Raised when location or annual climate data cannot be retrieved."""


@dataclass(frozen=True)
class AnnualWeather:
    rainfall_mm: float
    temperature_celsius: float
    resolved_location: str
    latitude: float
    longitude: float
    models: Tuple[str, ...]


class WeatherFetcher:
    """Fetches climate-model annual estimates from Open-Meteo."""

    def fetch_annual_weather(
        self, area: str, year: int, location: Optional[str] = None
    ) -> AnnualWeather:
        if not 1950 <= year <= 2049:
            raise ValueError("Climate data is available for years 1950 through 2049.")

        query = f"{location.strip()}, {area}" if location and location.strip() else area
        geocoded = self._get_json(
            GEOCODING_URL,
            {"name": query, "count": 1, "language": "en", "format": "json"},
            "location lookup",
        )
        results = geocoded.get("results")
        if not isinstance(results, list) or not results:
            raise WeatherFetchError(f"Could not find a location for {query!r}.")

        place = results[0]
        if not isinstance(place, dict):
            raise WeatherFetchError("The location service returned invalid location data.")
        country = place.get("country")
        if (
            location
            and location.strip()
            and (not isinstance(country, str) or country.casefold() != area.casefold())
        ):
            raise WeatherFetchError(
                f"The location {location.strip()!r} could not be verified in {area}."
            )

        try:
            latitude = float(place["latitude"])
            longitude = float(place["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherFetchError("The location service returned invalid coordinates.") from exc
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise WeatherFetchError("The location service returned out-of-range coordinates.")

        climate = self._get_json(
            CLIMATE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "models": ",".join(CLIMATE_MODELS),
                "daily": "temperature_2m_mean,precipitation_sum",
            },
            "climate data",
        )
        daily = climate.get("daily")
        if not isinstance(daily, dict):
            raise WeatherFetchError("The climate service returned no daily data.")

        expected_days = 366 if calendar.isleap(year) else 365
        dates = daily.get("time")
        if not isinstance(dates, list) or len(dates) != expected_days:
            raise WeatherFetchError(
                f"The climate service returned incomplete daily data for {year}."
            )

        annual_rainfall: List[float] = []
        annual_temperature: List[float] = []
        used_models: List[str] = []
        for model in CLIMATE_MODELS:
            rainfall = self._valid_values(daily.get(f"precipitation_sum_{model}"))
            temperature = self._valid_values(daily.get(f"temperature_2m_mean_{model}"))
            if len(rainfall) != expected_days or len(temperature) != expected_days:
                continue
            annual_rainfall.append(sum(rainfall))
            annual_temperature.append(sum(temperature) / expected_days)
            used_models.append(model)

        if not annual_rainfall or not annual_temperature:
            raise WeatherFetchError(
                f"The climate service returned no complete model estimates for {year}."
            )

        return AnnualWeather(
            rainfall_mm=round(median(annual_rainfall), 1),
            temperature_celsius=round(median(annual_temperature), 1),
            resolved_location=str(place.get("name", query)),
            latitude=latitude,
            longitude=longitude,
            models=tuple(used_models),
        )

    @staticmethod
    def _valid_values(values: Any) -> List[float]:
        if not isinstance(values, list):
            return []
        return [
            float(value)
            for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ]

    @staticmethod
    def _get_json(url: str, params: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise WeatherFetchError(f"Open-Meteo {service_name} request failed: {exc}") from exc
        except ValueError as exc:
            raise WeatherFetchError(
                f"Open-Meteo {service_name} response was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise WeatherFetchError(f"Open-Meteo returned an invalid {service_name} response.")
        return payload
