"""Tests for cleaner and feature engineering."""

import numpy as np
import pandas as pd

from src.preprocessing.cleaner import clean_combined_df
from src.preprocessing.feature_engineering import engineer_features, get_feature_columns


def _make_df(n=400):
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "price_eur_mwh": rng.uniform(20, 150, n),
        "load_mwh": rng.uniform(40_000, 80_000, n),
    })


class TestCleaner:
    def test_output_is_hourly(self):
        df = clean_combined_df(_make_df(100))
        deltas = pd.to_datetime(df["timestamp"], utc=True).diff().dropna()
        assert (deltas == pd.Timedelta("1h")).all()

    def test_deduplication(self):
        df = _make_df(50)
        doubled = pd.concat([df, df.head(10)], ignore_index=True)
        result = clean_combined_df(doubled)
        assert result["timestamp"].nunique() == len(result)

    def test_empty_returns_empty(self):
        df = pd.DataFrame({"timestamp": [], "price_eur_mwh": [], "load_mwh": []})
        assert clean_combined_df(df).empty

    def test_short_gaps_interpolated(self):
        df = _make_df(50)
        df.loc[5:7, "price_eur_mwh"] = np.nan
        result = clean_combined_df(df)
        assert result["price_eur_mwh"].isna().sum() == 0


class TestFeatureEngineering:
    def _cleaned(self, n=400):
        return clean_combined_df(_make_df(n))

    def test_all_feature_cols_present(self):
        df = engineer_features(self._cleaned(400))
        missing = [c for c in get_feature_columns() if c not in df.columns]
        assert not missing, f"Missing: {missing}"

    def test_hour_range(self):
        df = engineer_features(self._cleaned(400))
        assert df["hour"].between(0, 23).all()

    def test_is_weekend_binary(self):
        df = engineer_features(self._cleaned(400))
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_warmup_rows_dropped(self):
        df = engineer_features(self._cleaned(400))
        assert len(df) < 400

    def test_cyclical_bounds(self):
        df = engineer_features(self._cleaned(400))
        assert df["hour_sin"].between(-1.01, 1.01).all()
