"""Model versioning and persistence."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.utils.config import config
from src.utils.exceptions import ModelNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

MODEL_FILE = "electricity_model.joblib"
META_FILE = "model_metadata.json"


class ModelManager:
    def __init__(self, model_dir: Path | None = None) -> None:
        self._dir = model_dir or config.model.model_path
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self._dir / MODEL_FILE

    @property
    def meta_path(self) -> Path:
        return self._dir / META_FILE

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def generate_version(self) -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    def save(self, model: object, metadata: dict) -> str:
        version = metadata.get("version") or self.generate_version()
        metadata["version"] = version
        metadata["saved_at"] = datetime.now(tz=timezone.utc).isoformat()

        # Versioned backup
        backup = self._dir / f"electricity_model_v{version}.joblib"
        joblib.dump(model, backup)
        # Latest
        joblib.dump(model, self.model_path)
        # Metadata
        self.meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Model saved (version=%s).", version)
        return version

    def load(self) -> object:
        if not self.model_exists():
            raise ModelNotFoundError(f"No model at {self.model_path}. Train first.")
        model = joblib.load(self.model_path)
        logger.info("Model loaded from %s.", self.model_path)
        return model

    def load_metadata(self) -> dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())

    def get_latest_version(self) -> str | None:
        return self.load_metadata().get("version")
