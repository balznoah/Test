"""Validation for raw SMARD data — never raises on empty DataFrames."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

PRICE_MIN = -500.0
PRICE_MAX = 3000.0
LOAD_MIN = 10_000.0
LOAD_MAX = 120_000.0


def validate_price_series(df: pd.DataFrame) -> pd.DataFrame:
    """Validate price DataFrame. Returns empty DataFrame if input is empty."""
    if df.empty:
        logger.info("Price series is empty — nothing to validate.")
        return df
    df = _ensure_utc(df.copy())
    df = _drop_duplicates(df)
    mask = (df["price_eur_mwh"] < PRICE_MIN) | (df["price_eur_mwh"] > PRICE_MAX)
    if mask.any():
        logger.warning("%d price outliers set to NaN", mask.sum())
        df.loc[mask, "price_eur_mwh"] = float("nan")
    return df


def validate_load_series(df: pd.DataFrame) -> pd.DataFrame:
    """Validate load DataFrame. Returns empty DataFrame if input is empty."""
    if df.empty:
        logger.info("Load series is empty — nothing to validate.")
        return df
    df = _ensure_utc(df.copy())
    df = _drop_duplicates(df)
    mask = (df["load_mwh"] < LOAD_MIN) | (df["load_mwh"] > LOAD_MAX)
    if mask.any():
        logger.warning("%d load outliers set to NaN", mask.sum())
        df.loc[mask, "load_mwh"] = float("nan")
    return df


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < n:
        logger.info("Dropped %d duplicate rows", n - len(df))
    return df
