"""Clean and resample raw electricity data to uniform 1-hour intervals."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_INTERP_HOURS = 3


def clean_combined_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 1h, interpolate short gaps, drop all-NaN rows.

    Returns DataFrame with columns [timestamp, price_eur_mwh, load_mwh].
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure UTC datetime index
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    df = (
        df.drop_duplicates(subset=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )

    # Resample to 1h (handles quarter-hour SMARD data)
    df = df.resample("1h").mean()

    # Interpolate short gaps
    df["price_eur_mwh"] = df["price_eur_mwh"].interpolate(
        method="time", limit=MAX_INTERP_HOURS
    )
    df["load_mwh"] = df["load_mwh"].interpolate(
        method="time", limit=MAX_INTERP_HOURS
    )

    # Forward-fill remaining leading NaNs
    df = df.ffill(limit=6)

    before = len(df)
    df = df.dropna(how="all")
    logger.info("Cleaner: %d rows (%d all-NaN dropped).", len(df), before - len(df))
    return df.reset_index()
