"""SMARD API client with retry logic."""

import time
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import SmardConfig, config
from src.utils.exceptions import DataFetchError, NetworkError, ParseError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

INDEX_URL = "https://www.smard.de/app/chart_data/{fid}/{region}/index_{res}.json"
DATA_URL = "https://www.smard.de/app/chart_data/{fid}/{region}/{fid}_{region}_{res}_{ts}.json"


class SmardClient:
    """Fetches electricity price and load data from the SMARD public API."""

    def __init__(self, cfg: SmardConfig | None = None) -> None:
        self._cfg = cfg or config.smard
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "ElectricityForecast/1.0"

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str) -> Any:
        try:
            r = self._session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {url}") from e
        except requests.Timeout as e:
            raise NetworkError(f"Timeout: {url}") from e
        except requests.HTTPError as e:
            raise DataFetchError(f"HTTP {r.status_code}: {url}") from e
        except ValueError as e:
            raise ParseError(f"Invalid JSON: {url}") from e

    def get_timestamps(self, filter_id: int) -> list[int]:
        url = INDEX_URL.format(
            fid=filter_id, region=self._cfg.region, res=self._cfg.resolution
        )
        data = self._get(url)
        return sorted(data.get("timestamps", []))

    def get_chunk(self, filter_id: int, ts: int) -> list[tuple[datetime, float | None]]:
        url = DATA_URL.format(
            fid=filter_id, region=self._cfg.region, res=self._cfg.resolution, ts=ts
        )
        data = self._get(url)
        result = []
        for row in data.get("series", []):
            if len(row) == 2:
                dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
                result.append((dt, row[1]))
        return result

    def fetch(
        self, filter_id: int, since_ms: int | None = None
    ) -> list[tuple[datetime, float | None]]:
        """Fetch all data for a filter, optionally starting from since_ms."""
        all_ts = self.get_timestamps(filter_id)
        if since_ms is not None:
            # Keep only chunks that could contain newer data
            # Each chunk covers ~1 week; keep chunks from 2 weeks before since
            buffer_ms = 14 * 24 * 3600 * 1000
            all_ts = [t for t in all_ts if t >= since_ms - buffer_ms]

        logger.info("Fetching %d chunks for filter %d", len(all_ts), filter_id)
        rows: list[tuple[datetime, float | None]] = []
        for chunk_ts in all_ts:
            try:
                chunk = self.get_chunk(filter_id, chunk_ts)
                # Filter to only rows >= since_ms if specified
                if since_ms is not None:
                    chunk = [(dt, v) for dt, v in chunk
                             if int(dt.timestamp() * 1000) >= since_ms]
                rows.extend(chunk)
                time.sleep(0.1)
            except (DataFetchError, NetworkError, ParseError) as e:
                logger.warning("Skipping chunk %d: %s", chunk_ts, e)

        rows.sort(key=lambda x: x[0])
        logger.info("Fetched %d records for filter %d", len(rows), filter_id)
        return rows

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "SmardClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
