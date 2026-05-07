"""Model training with time-series cross-validation."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.database.repository import MetricsRepository
from src.ml.evaluator import compute_metrics
from src.ml.model_manager import ModelManager
from src.preprocessing.dataset_builder import DatasetBuilder
from src.utils.config import config
from src.utils.exceptions import ModelError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False


def _make_estimator():
    if _XGB:
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.model.random_state,
            n_jobs=-1,
            verbosity=0,
        )
    logger.warning("XGBoost not available, using GradientBoostingRegressor.")
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=config.model.random_state,
    )


class ModelTrainer:
    def __init__(
        self,
        builder: DatasetBuilder | None = None,
        manager: ModelManager | None = None,
    ) -> None:
        self._builder = builder or DatasetBuilder()
        self._manager = manager or ModelManager()

    def train(self) -> dict:
        logger.info("Building training dataset…")
        X, y, _ = self._builder.build_training_dataset()

        if len(X) < 300:
            raise ModelError(f"Too little data to train: {len(X)} rows (need ≥ 300).")

        tscv = TimeSeriesSplit(n_splits=config.model.n_splits)
        fold_metrics = []

        logger.info("Running %d-fold time-series CV…", config.model.n_splits)
        for fold, (tr, val) in enumerate(tscv.split(X), 1):
            pipe = Pipeline([("sc", StandardScaler()), ("m", _make_estimator())])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            m = compute_metrics(y.iloc[val], pipe.predict(X.iloc[val]))
            fold_metrics.append(m)
            logger.info("Fold %d — MAE=%.2f RMSE=%.2f MAPE=%.1f%%",
                        fold, m["mae"], m["rmse"], m.get("mape", float("nan")))

        avg = {
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "mape": float(np.nanmean([m.get("mape", float("nan")) for m in fold_metrics])),
        }
        logger.info("CV avg — MAE=%.2f RMSE=%.2f MAPE=%.1f%%",
                    avg["mae"], avg["rmse"], avg["mape"])

        # Final fit on all data
        logger.info("Final training on %d rows…", len(X))
        final = Pipeline([("sc", StandardScaler()), ("m", _make_estimator())])
        final.fit(X, y)

        version = self._manager.generate_version()
        self._manager.save(final, {
            "version": version,
            "train_rows": len(X),
            "cv_mae": avg["mae"],
            "cv_rmse": avg["rmse"],
            "cv_mape": avg["mape"],
            "features": list(X.columns),
        })

        MetricsRepository().save_metrics(
            version=version,
            mae=avg["mae"],
            rmse=avg["rmse"],
            mape=avg["mape"],
            train_rows=len(X),
        )

        logger.info("Training complete. Version: %s", version)
        return {"version": version, **avg}
