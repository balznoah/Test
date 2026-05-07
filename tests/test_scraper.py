"""Tests for SMARD client and validators."""

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from src.scraper.smard_client import SmardClient
from src.scraper.validators import validate_load_series, validate_price_series
from src.utils.exceptions import DataFetchError


class TestSmardClient:
    @resp_lib.activate
    def test_timestamps_sorted(self):
        resp_lib.add(resp_lib.GET,
                     "https://www.smard.de/app/chart_data/4169/DE/index_hour.json",
                     json={"timestamps": [3000, 1000, 2000]})
        assert SmardClient().get_timestamps(4169) == [1000, 2000, 3000]

    @resp_lib.activate
    def test_empty_timestamps(self):
        resp_lib.add(resp_lib.GET,
                     "https://www.smard.de/app/chart_data/4169/DE/index_hour.json",
                     json={"timestamps": []})
        assert SmardClient().get_timestamps(4169) == []

    @resp_lib.activate
    def test_http_error_raises(self):
        resp_lib.add(resp_lib.GET,
                     "https://www.smard.de/app/chart_data/4169/DE/index_hour.json",
                     status=500)
        with pytest.raises(DataFetchError):
            SmardClient().get_timestamps(4169)


class TestValidators:
    def _price_df(self, prices):
        ts = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "price_eur_mwh": prices})

    def _load_df(self, loads):
        ts = pd.date_range("2024-01-01", periods=len(loads), freq="1h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "load_mwh": loads})

    def test_valid_prices_pass(self):
        assert len(validate_price_series(self._price_df([50, 80, 120]))) == 3

    def test_outlier_becomes_nan(self):
        df = validate_price_series(self._price_df([50, 9999, 80]))
        assert pd.isna(df.loc[1, "price_eur_mwh"])

    def test_empty_returns_empty(self):
        df = pd.DataFrame({"timestamp": [], "price_eur_mwh": []})
        assert validate_price_series(df).empty

    def test_duplicates_dropped(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({"timestamp": [ts, ts], "price_eur_mwh": [50.0, 60.0]})
        assert len(validate_price_series(df)) == 1

    def test_empty_load_returns_empty(self):
        df = pd.DataFrame({"timestamp": [], "load_mwh": []})
        assert validate_load_series(df).empty

    def test_load_outlier_becomes_nan(self):
        df = validate_load_series(self._load_df([50_000, -1, 60_000]))
        assert pd.isna(df.loc[1, "load_mwh"])
