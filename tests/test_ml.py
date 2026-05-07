"""Tests for evaluator and model manager."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.ml.evaluator import compute_metrics
from src.ml.model_manager import ModelManager
from src.utils.exceptions import ModelNotFoundError


class TestMetrics:
    def test_perfect(self):
        y = pd.Series([10.0, 20.0, 30.0])
        m = compute_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)

    def test_mae_correct(self):
        m = compute_metrics(pd.Series([10, 20, 30]), pd.Series([12, 18, 33]))
        assert m["mae"] == pytest.approx((2 + 2 + 3) / 3, rel=1e-5)

    def test_mape_nonzero(self):
        m = compute_metrics(pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0]))
        assert m["mape"] == pytest.approx(((10/100) + (20/200)) / 2 * 100, rel=1e-3)

    def test_mape_zero_actuals(self):
        m = compute_metrics(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))
        assert np.isnan(m["mape"])


class TestModelManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        return ModelManager(model_dir=tmp_path)

    def test_not_exists_initially(self, mgr):
        assert not mgr.model_exists()

    def test_load_raises_if_missing(self, mgr):
        with pytest.raises(ModelNotFoundError):
            mgr.load()

    def test_save_load_roundtrip(self, mgr):
        from sklearn.linear_model import LinearRegression
        mgr.save(LinearRegression(), {"version": "v1"})
        assert mgr.model_exists()
        assert isinstance(mgr.load(), LinearRegression)

    def test_metadata_persisted(self, mgr):
        from sklearn.linear_model import LinearRegression
        mgr.save(LinearRegression(), {"version": "v42", "cv_mae": 3.14})
        meta = mgr.load_metadata()
        assert meta["version"] == "v42"
        assert meta["cv_mae"] == pytest.approx(3.14)

    def test_versioned_backup_created(self, mgr):
        from sklearn.linear_model import LinearRegression
        mgr.save(LinearRegression(), {"version": "backup_test"})
        backups = list(mgr._dir.glob("electricity_model_v*.joblib"))
        assert len(backups) == 1


class TestTrainerSmoke:
    def test_end_to_end(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from src.ml.trainer import ModelTrainer
        from src.preprocessing.feature_engineering import get_feature_columns

        n = 500
        cols = get_feature_columns()
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.uniform(0, 1, (n, len(cols))), columns=cols)
        y = pd.Series(rng.uniform(20, 150, n))

        mock_builder = MagicMock()
        mock_builder.build_training_dataset.return_value = (
            X, y, pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        )
        mgr = ModelManager(model_dir=tmp_path / "models")

        with patch("src.database.repository.MetricsRepository.save_metrics"):
            trainer = ModelTrainer(builder=mock_builder, manager=mgr)
            result = trainer.train()

        assert "version" in result
        assert result["mae"] >= 0
        assert mgr.model_exists()
