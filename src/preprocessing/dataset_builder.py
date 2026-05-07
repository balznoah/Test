"""Builds training and inference datasets from the database."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.database.repository import ElectricityRepository
from src.preprocessing.cleaner import clean_combined_df
from src.preprocessing.feature_engineering import engineer_features, get_feature_columns
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

TARGET = "price_eur_mwh"


class DatasetBuilder:
    def __init__(self, repo: ElectricityRepository | None = None) -> None:
        self._repo = repo or ElectricityRepository()
        self._feat_cols = get_feature_columns()

    def build_training_dataset(self) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        raw = self._repo.get_all_records()
        if raw.empty:
            raise ValueError("No data in database. Run ingestion first.")

        cleaned = clean_combined_df(raw)
        featured = engineer_features(cleaned)

        cols = [c for c in self._feat_cols if c in featured.columns]
        X = featured[cols].copy()
        y = featured[TARGET].copy()
        timestamps = pd.DatetimeIndex(featured["timestamp"])

        logger.info("Training set: %d rows × %d features.", len(X), len(cols))
        return X, y, timestamps

    def build_inference_input(
        self, horizon_hours: int | None = None
    ) -> tuple[pd.DataFrame, list[datetime]]:
        """Build feature matrix for the next N hours."""
        horizon = horizon_hours or config.model.forecast_horizon_hours

        # Use last 21 days of data to cover all lag windows (max lag = 168h = 7 days)
        since = datetime.now(tz=timezone.utc) - timedelta(days=21)
        raw = self._repo.get_records_since(since)

        if raw.empty:
            raise ValueError("No recent data for inference.")

        cleaned = clean_combined_df(raw)

        last_ts = pd.to_datetime(cleaned["timestamp"], utc=True).max()
        future_ts = [last_ts + timedelta(hours=h) for h in range(1, horizon + 1)]

        # Append future rows with NaN targets so lag features can be computed
        future_df = pd.DataFrame({
            "timestamp": future_ts,
            "price_eur_mwh": float("nan"),
            "load_mwh": float("nan"),
        })
        extended = pd.concat([cleaned, future_df], ignore_index=True).sort_values("timestamp")

        featured = engineer_features(extended)

        # Select only future rows
        future_mask = featured["timestamp"].isin(future_ts)
        future_feat = featured[future_mask].reset_index(drop=True)

        cols = [c for c in self._feat_cols if c in future_feat.columns]
        X_future = future_feat[cols].copy()

        # Fill any remaining NaNs forward from historical context
        X_future = X_future.ffill().bfill()

        logger.info("Inference input: %d rows × %d features.", len(X_future), len(cols))
        return X_future, future_ts
