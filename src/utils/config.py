"""Central configuration loaded from environment / .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SmardConfig:
    base_url: str = "https://www.smard.de/app/chart_data"
    price_filter: int = 4169   # Day-Ahead Preis DE/LU
    load_filter: int = 410     # Realisierter Stromverbrauch
    region: str = "DE"
    resolution: str = "hour"


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/electricity.db")
    )
    echo: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", "models"))
    )
    forecast_horizon_hours: int = 24
    random_state: int = 42
    n_splits: int = 5


@dataclass(frozen=True)
class EmailConfig:
    gmail_user: str = field(default_factory=lambda: os.getenv("GMAIL_USER", ""))
    gmail_password: str = field(default_factory=lambda: os.getenv("GMAIL_PASSWORD", ""))
    email_receiver: str = field(default_factory=lambda: os.getenv("EMAIL_RECEIVER", ""))
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    def is_configured(self) -> bool:
        return bool(self.gmail_user and self.gmail_password and self.email_receiver)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("data/reports")
    historical_days: int = 90


@dataclass(frozen=True)
class AppConfig:
    smard: SmardConfig = field(default_factory=SmardConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def ensure_directories(self) -> None:
        for d in [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.reports_dir,
            self.model.model_path,
            Path("logs"),
        ]:
            d.mkdir(parents=True, exist_ok=True)


config = AppConfig()
