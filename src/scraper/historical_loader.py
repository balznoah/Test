"""Loads and merges price + load data from SMARD."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.scraper.smard_client import SmardClient
from src.scraper.validators import validate_load_series, validate_price_series
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)


class HistoricalLoader:
    def __init__(self, client: SmardClient | None = None) -> None:
        self._client = client or SmardClient()

    def load_combined(self, since: datetime | None = None) -> pd.DataFrame:
        """
        Fetch price + load data and merge them.

        Args:
            since: Only fetch data after this UTC datetime.
                   If None, fetches the last `historical_days` days.

        Returns:
            DataFrame with columns [timestamp, price_eur_mwh, load_mwh].
            May be empty if no new data is available.
        """
        since_ms = self._resolve_since_ms(since)
        logger.info(
            "Fetching data since %s",
            datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat(),
        )

        price_rows = self._client.fetch(config.smard.price_filter, since_ms)
        load_rows = self._client.fetch(config.smard.load_filter, since_ms)

        price_df = pd.DataFrame(price_rows, columns=["timestamp", "price_eur_mwh"])
        load_df = pd.DataFrame(load_rows, columns=["timestamp", "load_mwh"])

        price_df = validate_price_series(price_df)
        load_df = validate_load_series(load_df)

        if price_df.empty and load_df.empty:
            logger.info("No new data available from SMARD.")
            return pd.DataFrame(columns=["timestamp", "price_eur_mwh", "load_mwh"])

        # Merge on timestamp
        if price_df.empty:
            combined = load_df
            combined["price_eur_mwh"] = float("nan")
        elif load_df.empty:
            combined = price_df
            combined["load_mwh"] = float("nan")
        else:
            price_df = price_df.set_index("timestamp")
            load_df = load_df.set_index("timestamp")
            combined = price_df.join(load_df, how="outer").sort_index()
            combined = combined.dropna(how="all").reset_index()

        logger.info("Combined dataset: %d rows", len(combined))
        return combined

    def _resolve_since_ms(self, since: datetime | None) -> int:
        if since is not None:
            # Ensure UTC-aware
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            return int(since.timestamp() * 1000)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=config.data.historical_days)
        return int(cutoff.timestamp() * 1000)
