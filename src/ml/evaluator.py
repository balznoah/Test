"""Regression metrics: MAE, RMSE, MAPE."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    nonzero = np.abs(y_true) > 5.0  # exclude near-zero/negative prices from MAPE
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.sum() > 0 else float("nan")

    logger.info("MAE=%.3f  RMSE=%.3f  MAPE=%.2f%%", mae, rmse, mape)
    return {"mae": mae, "rmse": rmse, "mape": mape}
