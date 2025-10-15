"""Utility helpers for working with real F1 weather data.

This module purposefully implements only the subset of the original
``f1_ml.weather`` API that is required by
``notebooks/advanced/fetch_weather_data.py``.  The historic implementation
that used to live in ``archive/old_pipeline`` provided a very feature rich
``F1WeatherProvider`` class, however re-introducing ~600 lines of legacy code
into the active tree made maintenance difficult.  The goal of this file is to
offer a focused, well-tested provider that keeps backward compatibility with
the notebook tooling while delegating most heavy lifting to the Visual
Crossing API.

The provider exposes three key behaviours used by the notebook script:

* Loading previously cached results from ``data/weather_cache``
* Checking whether a circuit/date/session combination already exists in the
  CSV cache
* Fetching and caching fresh weather data when needed

The implementation below keeps the public method surface identical to the
original module (``get_weather_for_race`` and ``_check_csv_cache`` together
with the ``csv_cache_files`` attribute).  Internally it favours clarity and
robust error handling over legacy compatibility shims.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class F1WeatherProvider:
    """Fetch and cache weather information for F1 sessions."""

    #: Default CSV file names per session type used by the legacy notebook
    _SESSION_CACHE_FILES = {
        "race": "f1_weather_data.csv",
        "sprint": "f1_weather_data_sprint.csv",
        "qualifying": "f1_weather_data_qualifying.csv",
        "practice": "f1_weather_data_practice.csv",
    }

    #: Canonical set of columns stored inside the CSV cache files
    _CACHE_COLUMNS: Tuple[str, ...] = (
        "circuit_name",
        "date",
        "session_type",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation",
        "rain_probability",
        "conditions",
        "description",
        "is_wet_race",
        "track_temp",
        "weather_changeability",
        "provider",
        "fetched_at",
    )

    #: Minimal set of circuit coordinates required by Visual Crossing lookups
    _CIRCUIT_COORDINATES = {
        "bahrain": (26.0325, 50.5106),
        "jeddah": (21.6319, 39.1044),
        "albert_park": (-37.8497, 144.9680),
        "shanghai": (31.3389, 121.2198),
        "miami": (25.9581, -80.2389),
        "imola": (44.3439, 11.7167),
        "monaco": (43.7347, 7.4144),
        "catalunya": (41.5700, 2.2611),
        "villeneuve": (45.5000, -73.5228),
        "red_bull_ring": (47.2197, 14.7647),
        "silverstone": (52.0786, -1.0169),
        "hungaroring": (47.5789, 19.2486),
        "spa": (50.4372, 5.9714),
        "zandvoort": (52.3888, 4.5409),
        "monza": (45.6156, 9.2811),
        "marina_bay": (1.2914, 103.8644),
        "americas": (30.1328, -97.6411),
        "rodriguez": (19.4042, -99.0907),
        "interlagos": (-23.7036, -46.6997),
        "las_vegas": (36.1147, -115.1730),
        "yas_marina": (24.4672, 54.6031),
        "baku": (40.3725, 49.8533),
        "losail": (25.4900, 51.4542),
        "suzuka": (34.8431, 136.5411),
        "sepang": (2.7606, 101.7381),
        "hockenheim": (49.3278, 8.5656),
        "fuji": (35.3717, 138.9256),
    }

    def __init__(self, api_key: Optional[str] = None, provider: str = "visual_crossing") -> None:
        if provider != "visual_crossing":  # pragma: no cover - guard rail for legacy imports
            raise ValueError(
                "Only the Visual Crossing provider is currently supported."
            )

        self.provider = provider
        self.api_key = api_key or os.environ.get("VISUAL_CROSSING_API_KEY")

        cache_root = Path(__file__).resolve().parents[3] / "data" / "weather_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_root

        self.csv_cache_files = {
            session: cache_root / filename
            for session, filename in self._SESSION_CACHE_FILES.items()
        }

        # Lazily materialise cache data frames
        self._cache_frames: Dict[str, pd.DataFrame] = {}

        # Allow callers to extend the coordinates dictionary as needed
        self.circuit_coordinates = dict(self._CIRCUIT_COORDINATES)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cache_frame(self, session_type: str) -> pd.DataFrame:
        """Return the cache DataFrame for ``session_type``.

        The notebook frequently inspects the cache to avoid unnecessary API
        calls.  We keep the DataFrame in memory for the lifetime of the
        provider and re-load it from disk on demand.
        """

        if session_type not in self._SESSION_CACHE_FILES:
            raise ValueError(f"Unknown session type: {session_type}")

        if session_type not in self._cache_frames:
            cache_file = self.csv_cache_files[session_type]
            if cache_file.exists():
                try:
                    frame = pd.read_csv(cache_file)
                except Exception as exc:  # pragma: no cover - defensive branch
                    logger.warning(
                        "Failed to load weather cache %s (%s); starting with an empty frame",
                        cache_file,
                        exc,
                    )
                    frame = pd.DataFrame(columns=self._CACHE_COLUMNS)
            else:
                frame = pd.DataFrame(columns=self._CACHE_COLUMNS)

            # Ensure required columns exist even if the CSV was created by an
            # older version of the tooling.
            for column in self._CACHE_COLUMNS:
                if column not in frame.columns:
                    frame[column] = pd.Series(dtype="object")

            self._cache_frames[session_type] = frame

        return self._cache_frames[session_type]

    def _normalise_circuit_key(self, circuit_name: str) -> str:
        return (
            circuit_name.strip().lower().replace(" ", "_").replace("-", "_")
        )

    def _check_csv_cache(self, circuit_name: str, date: str, session_type: str = "race") -> bool:
        """Return ``True`` if a cached entry exists for ``circuit_name``."""

        frame = self._get_cache_frame(session_type)
        if frame.empty:
            return False

        mask = (
            frame["circuit_name"].astype(str).str.lower() == circuit_name.lower()
        ) & (frame["date"].astype(str) == str(date))
        return bool(mask.any())

    # The notebook accesses this method directly, so we keep it public.
    _check_csv_cache.__doc__ = "Check whether weather data is cached."  # type: ignore[attr-defined]

    def _append_to_cache(self, session_type: str, record: Dict[str, object]) -> None:
        frame = self._get_cache_frame(session_type)
        updated = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
        updated = updated.drop_duplicates(subset=["circuit_name", "date"], keep="last")
        self._cache_frames[session_type] = updated

        cache_file = self.csv_cache_files[session_type]
        updated.to_csv(cache_file, index=False)

    # ------------------------------------------------------------------
    # Weather fetching
    # ------------------------------------------------------------------
    def get_circuit_coordinates(self, circuit_name: str) -> Optional[Tuple[float, float]]:
        """Return latitude/longitude for ``circuit_name`` if known."""

        key = self._normalise_circuit_key(circuit_name)
        return self.circuit_coordinates.get(key)

    def _visual_crossing_location(self, _: str, lat: float, lon: float) -> str:
        """Visual Crossing accepts the ``lat,lon`` pair as a location string."""

        return f"{lat:.4f},{lon:.4f}"

    def _fetch_visual_crossing_weather(self, circuit_name: str, date: str) -> Dict[str, object]:
        if not self.api_key:
            raise RuntimeError(
                "No Visual Crossing API key configured. Set the VISUAL_CROSSING_API_KEY "
                "environment variable or pass api_key= when constructing F1WeatherProvider."
            )

        coordinates = self.get_circuit_coordinates(circuit_name)
        if not coordinates:
            raise ValueError(f"No coordinates available for circuit '{circuit_name}'.")

        lat, lon = coordinates
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        location = self._visual_crossing_location(circuit_name, lat, lon)

        params = {
            "unitGroup": "metric",
            "key": self.api_key,
            "include": "hours",
            "contentType": "json",
        }

        url = f"{base_url}/{location}/{date}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        day_data = payload.get("days", [{}])[0]
        hourly = day_data.get("hours", [{}])
        midday = hourly[min(len(hourly) // 2, len(hourly) - 1)]

        return {
            "temperature": float(day_data.get("temp", midday.get("temp", 0.0))),
            "humidity": float(day_data.get("humidity", midday.get("humidity", 0.0))),
            "wind_speed": float(day_data.get("windspeed", midday.get("windspeed", 0.0))),
            "precipitation": float(day_data.get("precip", midday.get("precip", 0.0))),
            "rain_probability": float(day_data.get("precipprob", midday.get("precipprob", 0.0))) / 100.0,
            "conditions": day_data.get("conditions", "Unknown"),
            "description": day_data.get("description", ""),
            "is_wet_race": bool(day_data.get("precip", 0) or day_data.get("precipprob", 0) > 50),
            "track_temp": float(midday.get("temp", day_data.get("temp", 0.0))) + 10.0,
            "weather_changeability": float(day_data.get("cloudcover", 50)) / 100.0,
        }

    def _store_json_cache(self, circuit_name: str, date: str, payload: Dict[str, object]) -> None:
        safe_name = self._normalise_circuit_key(circuit_name)
        json_path = self.cache_dir / f"visual_crossing_{safe_name}_{date}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def get_weather_for_race(
        self,
        circuit_name: str,
        date: str,
        session_type: str = "race",
        force_refresh: bool = False,
    ) -> Dict[str, object]:
        """Return weather information for a given race session.

        Parameters
        ----------
        circuit_name:
            Human readable circuit name (e.g. ``"Silverstone"``).
        date:
            Session date in ``YYYY-MM-DD`` format.
        session_type:
            One of ``race``, ``sprint``, ``qualifying`` or ``practice``.  The
            notebook keeps separate caches per session type and expects the
            provider to follow suit.
        force_refresh:
            If ``True`` skip the cache and fetch data from the API again.
        """

        session_type = session_type or "race"
        if not force_refresh and self._check_csv_cache(circuit_name, date, session_type):
            frame = self._get_cache_frame(session_type)
            cached_row = frame[
                (frame["circuit_name"].astype(str).str.lower() == circuit_name.lower())
                & (frame["date"].astype(str) == str(date))
            ].iloc[0]
            return cached_row.to_dict()

        weather = self._fetch_visual_crossing_weather(circuit_name, date)
        record = {
            **weather,
            "circuit_name": circuit_name,
            "date": date,
            "session_type": session_type,
            "provider": self.provider,
            "fetched_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

        self._append_to_cache(session_type, record)
        self._store_json_cache(circuit_name, date, record)

        return record

    # ------------------------------------------------------------------
    # Bulk helpers retained for compatibility (used in docs/tests)
    # ------------------------------------------------------------------
    def get_weather_for_sessions(
        self, items: Iterable[Tuple[str, str, str]]
    ) -> Dict[Tuple[str, str, str], Dict[str, object]]:
        """Convenience wrapper that fetches multiple sessions in sequence."""

        results = {}
        for circuit_name, date, session_type in items:
            try:
                results[(circuit_name, date, session_type)] = self.get_weather_for_race(
                    circuit_name,
                    date,
                    session_type=session_type,
                )
            except Exception as exc:  # pragma: no cover - helper for manual use
                logger.warning(
                    "Failed to fetch weather for %s on %s (%s): %s",
                    circuit_name,
                    date,
                    session_type,
                    exc,
                )

        return results


__all__ = ["F1WeatherProvider"]

