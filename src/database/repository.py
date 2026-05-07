"""All database read/write operations."""

import math
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.database.db import get_session
from src.database.models import ElectricityRecord, ModelMetrics, Prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _nan_to_none(v):
    if v is None:
        return None
    try:
        return None if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


def _to_utc(ts):
    """Convert any timestamp to UTC-aware datetime."""
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


class ElectricityRepository:

    def upsert_records(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        now = datetime.now(tz=timezone.utc)
        records = [
            {
                "timestamp": _to_utc(row["timestamp"]),
                "price_eur_mwh": _nan_to_none(row.get("price_eur_mwh")),
                "load_mwh": _nan_to_none(row.get("load_mwh")),
                "created_at": now,
            }
            for _, row in df.iterrows()
        ]
        with get_session() as s:
            stmt = sqlite_insert(ElectricityRecord).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp"],
                set_={
                    "price_eur_mwh": stmt.excluded.price_eur_mwh,
                    "load_mwh": stmt.excluded.load_mwh,
                },
            )
            s.execute(stmt)
        logger.info("Upserted %d records.", len(records))
        return len(records)

    def get_latest_timestamp(self) -> datetime | None:
        with get_session() as s:
            result = s.execute(
                select(ElectricityRecord.timestamp)
                .order_by(ElectricityRecord.timestamp.desc())
                .limit(1)
            ).scalar_one_or_none()
        if result is None:
            return None
        return _to_utc(result)

    def get_records_since(self, since: datetime) -> pd.DataFrame:
        since = _to_utc(since)
        with get_session() as s:
            rows = s.execute(
                select(ElectricityRecord)
                .where(ElectricityRecord.timestamp >= since)
                .order_by(ElectricityRecord.timestamp)
            ).scalars().all()
            return pd.DataFrame(
                [{"timestamp": _to_utc(r.timestamp),
                  "price_eur_mwh": r.price_eur_mwh,
                  "load_mwh": r.load_mwh} for r in rows]
            )

    def get_all_records(self) -> pd.DataFrame:
        with get_session() as s:
            rows = s.execute(
                select(ElectricityRecord).order_by(ElectricityRecord.timestamp)
            ).scalars().all()
            return pd.DataFrame(
                [{"timestamp": _to_utc(r.timestamp),
                  "price_eur_mwh": r.price_eur_mwh,
                  "load_mwh": r.load_mwh} for r in rows]
            )


class PredictionRepository:

    def save_predictions(self, df: pd.DataFrame, model_version: str) -> int:
        if df.empty:
            return 0
        now = datetime.now(tz=timezone.utc)
        records = [
            {
                "forecast_timestamp": _to_utc(row["forecast_timestamp"]),
                "predicted_price_eur_mwh": float(row["predicted_price_eur_mwh"]),
                "model_version": model_version,
                "generated_at": now,
            }
            for _, row in df.iterrows()
        ]
        with get_session() as s:
            stmt = sqlite_insert(Prediction).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["forecast_timestamp", "model_version"],
                set_={"predicted_price_eur_mwh": stmt.excluded.predicted_price_eur_mwh,
                      "generated_at": stmt.excluded.generated_at},
            )
            s.execute(stmt)
        logger.info("Saved %d predictions (version=%s).", len(records), model_version)
        return len(records)

    def get_latest_predictions(self, model_version: str) -> pd.DataFrame:
        with get_session() as s:
            # Get latest generated_at for this version
            latest_gen = s.execute(
                select(Prediction.generated_at)
                .where(Prediction.model_version == model_version)
                .order_by(Prediction.generated_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            if latest_gen is None:
                return pd.DataFrame()

            rows = s.execute(
                select(Prediction)
                .where(Prediction.model_version == model_version)
                .where(Prediction.generated_at == latest_gen)
                .order_by(Prediction.forecast_timestamp)
            ).scalars().all()

            return pd.DataFrame(
                [{"forecast_timestamp": _to_utc(r.forecast_timestamp),
                  "predicted_price_eur_mwh": r.predicted_price_eur_mwh,
                  "model_version": r.model_version} for r in rows]
            )


class MetricsRepository:

    def save_metrics(self, version: str, mae: float, rmse: float,
                     mape: float | None, train_rows: int) -> None:
        now = datetime.now(tz=timezone.utc)
        with get_session() as s:
            stmt = sqlite_insert(ModelMetrics).values([{
                "model_version": version,
                "mae": mae, "rmse": rmse, "mape": mape,
                "train_rows": train_rows, "trained_at": now,
            }])
            stmt = stmt.on_conflict_do_update(
                index_elements=["model_version"],
                set_={"mae": stmt.excluded.mae, "rmse": stmt.excluded.rmse,
                      "mape": stmt.excluded.mape, "trained_at": stmt.excluded.trained_at},
            )
            s.execute(stmt)
        logger.info("Saved metrics for version %s.", version)
