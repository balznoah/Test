"""Tests for database repository layer."""

from contextlib import contextmanager
from datetime import timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database.repository import ElectricityRepository, PredictionRepository


@pytest.fixture
def patched_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    @contextmanager
    def _session():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr("src.database.repository.get_session", _session)


def _elec_df(n=5):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "price_eur_mwh": [50.0 + i for i in range(n)],
        "load_mwh": [60_000.0 + i * 100 for i in range(n)],
    })


class TestElectricityRepository:
    def test_upsert_inserts(self, patched_session):
        repo = ElectricityRepository()
        assert repo.upsert_records(_elec_df(5)) == 5

    def test_no_duplicates(self, patched_session):
        repo = ElectricityRepository()
        repo.upsert_records(_elec_df(5))
        repo.upsert_records(_elec_df(5))
        assert len(repo.get_all_records()) == 5

    def test_get_records_since(self, patched_session):
        repo = ElectricityRepository()
        repo.upsert_records(_elec_df(10))
        cutoff = pd.Timestamp("2024-01-01 05:00", tz="UTC")
        result = repo.get_records_since(cutoff)
        assert all(pd.to_datetime(result["timestamp"], utc=True) >= cutoff)

    def test_latest_timestamp(self, patched_session):
        repo = ElectricityRepository()
        repo.upsert_records(_elec_df(5))
        ts = repo.get_latest_timestamp()
        assert ts is not None
        assert ts.tzinfo is not None  # must be timezone-aware

    def test_empty_df_returns_zero(self, patched_session):
        assert ElectricityRepository().upsert_records(pd.DataFrame()) == 0


class TestPredictionRepository:
    def _fc_df(self, n=24):
        ts = pd.date_range("2024-01-02", periods=n, freq="1h", tz="UTC")
        return pd.DataFrame({
            "forecast_timestamp": ts,
            "predicted_price_eur_mwh": [75.0] * n,
        })

    def test_save_and_retrieve(self, patched_session):
        repo = PredictionRepository()
        repo.save_predictions(self._fc_df(24), "v1")
        result = repo.get_latest_predictions("v1")
        assert len(result) == 24

    def test_upsert_updates(self, patched_session):
        repo = PredictionRepository()
        repo.save_predictions(self._fc_df(3), "v1")
        updated = self._fc_df(3)
        updated["predicted_price_eur_mwh"] = 99.0
        repo.save_predictions(updated, "v1")
        result = repo.get_latest_predictions("v1")
        assert (result["predicted_price_eur_mwh"] == 99.0).all()
