"""Inference: load model, generate 24h forecast, persist to DB."""

import pandas as pd

from src.database.repository import PredictionRepository
from src.ml.model_manager import ModelManager
from src.preprocessing.dataset_builder import DatasetBuilder
from src.utils.config import config
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)


class Predictor:
    def __init__(
        self,
        builder: DatasetBuilder | None = None,
        manager: ModelManager | None = None,
        repo: PredictionRepository | None = None,
    ) -> None:
        self._builder = builder or DatasetBuilder()
        self._manager = manager or ModelManager()
        self._repo = repo or PredictionRepository()

    def predict(self, horizon_hours: int | None = None, persist: bool = True) -> pd.DataFrame:
        if not self._manager.model_exists():
            raise PredictionError("No model found. Run training first.")

        horizon = horizon_hours or config.model.forecast_horizon_hours
        model = self._manager.load()
        version = self._manager.get_latest_version() or "unknown"

        logger.info("Generating %d-hour forecast (model=%s).", horizon, version)

        try:
            X_future, future_ts = self._builder.build_inference_input(horizon_hours=horizon)
        except Exception as e:
            raise PredictionError(f"Inference input failed: {e}") from e

        try:
            preds = model.predict(X_future)
        except Exception as e:
            raise PredictionError(f"model.predict() failed: {e}") from e

        result = pd.DataFrame({
            "forecast_timestamp": future_ts,
            "predicted_price_eur_mwh": preds,
            "model_version": version,
        })

        logger.info(
            "Forecast stats: min=%.2f max=%.2f mean=%.2f EUR/MWh",
            result["predicted_price_eur_mwh"].min(),
            result["predicted_price_eur_mwh"].max(),
            result["predicted_price_eur_mwh"].mean(),
        )

        if persist:
            self._repo.save_predictions(result, model_version=version)

        return result
