"""Feature engineering for electricity price forecasting."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [6, 24, 168]
MAX_LAG = max(LAG_HOURS)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar, lag, rolling, and delta features.

    Input: DataFrame with [timestamp, price_eur_mwh, load_mwh]
    Output: DataFrame with all original columns + feature columns
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    ts = df["timestamp"]

    # Calendar
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    for lag in LAG_HOURS:
        df[f"price_lag_{lag}h"] = df["price_eur_mwh"].shift(lag)
        df[f"load_lag_{lag}h"] = df["load_mwh"].shift(lag)

    # Rolling statistics (using shifted series to avoid leakage)
    for w in ROLLING_WINDOWS:
        df[f"price_roll_mean_{w}h"] = (
            df["price_eur_mwh"].shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"price_roll_std_{w}h"] = (
            df["price_eur_mwh"].shift(1).rolling(w, min_periods=2).std().fillna(0)
        )
        df[f"load_roll_mean_{w}h"] = (
            df["load_mwh"].shift(1).rolling(w, min_periods=1).mean()
        )

    # Deltas
    df["price_delta_1h"] = df["price_eur_mwh"].diff(1)
    df["price_delta_24h"] = df["price_eur_mwh"].diff(24)
    df["load_delta_1h"] = df["load_mwh"].diff(1)

    before = len(df)
    # Drop warm-up rows where the longest lag is NaN
    df = df.dropna(subset=[f"price_lag_{MAX_LAG}h"]).reset_index(drop=True)
    logger.info("Features: %d → %d rows (%d warm-up dropped).", before, len(df), before - len(df))
    return df


def get_feature_columns() -> list[str]:
    cols = [
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "price_delta_1h", "price_delta_24h", "load_delta_1h",
    ]
    for lag in LAG_HOURS:
        cols += [f"price_lag_{lag}h", f"load_lag_{lag}h"]
    for w in ROLLING_WINDOWS:
        cols += [f"price_roll_mean_{w}h", f"price_roll_std_{w}h", f"load_roll_mean_{w}h"]
    return cols
