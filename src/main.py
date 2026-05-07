"""
ElectricityForecast pipeline orchestrator.

Steps:
  1. Data ingestion  — fetch/update from SMARD
  2. Model training  — train if no model exists, or --force-train
  3. Prediction      — generate 24h forecast
  4. Report          — CSV + HTML + charts
  5. Email           — send report if credentials configured
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

from src.database.db import get_engine, init_db
from src.database.repository import ElectricityRepository
from src.ml.model_manager import ModelManager
from src.ml.predictor import Predictor
from src.ml.trainer import ModelTrainer
from src.notifications.email_sender import EmailSender
from src.reporting.report_generator import ReportGenerator
from src.scraper.historical_loader import HistoricalLoader
from src.utils.config import config
from src.utils.exceptions import (
    ConfigurationError,
    DataFetchError,
    EmailError,
    ModelError,
    PredictionError,
    ReportError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)


# ── Step implementations ──────────────────────────────────────────────────────

def step_ingest() -> int:
    repo = ElectricityRepository()
    loader = HistoricalLoader()

    latest = repo.get_latest_timestamp()
    if latest is not None:
        # Ensure UTC-aware; back off 6h to catch any gaps
        since = latest - timedelta(hours=6)
        # Never request data from the future
        now = datetime.now(tz=timezone.utc)
        if since > now:
            since = now - timedelta(hours=6)
        logger.info("Incremental update since %s", since.isoformat())
    else:
        since = None
        logger.info("No existing data — initial historical load.")

    df = loader.load_combined(since=since)
    if df.empty:
        logger.info("No new data from SMARD — database is up to date.")
        return 0

    count = repo.upsert_records(df)
    logger.info("Ingested %d records.", count)
    return count


def step_train(force: bool = False) -> dict:
    manager = ModelManager()
    if manager.model_exists() and not force:
        meta = manager.load_metadata()
        logger.info("Model already exists (version=%s). Skipping training.", meta.get("version"))
        return meta
    return ModelTrainer().train()


def step_predict() -> None:
    result = Predictor().predict(persist=True)
    logger.info(
        "Generated %d predictions. Mean=%.2f EUR/MWh",
        len(result), result["predicted_price_eur_mwh"].mean()
    )


def step_report() -> dict[str, object]:
    return ReportGenerator().generate()


def step_email(report_paths: dict) -> None:
    sender = EmailSender()
    charts = [v for k, v in report_paths.items() if k not in ("csv", "html")]
    sender.send_daily_report(
        html_path=report_paths["html"],
        csv_path=report_paths["csv"],
        chart_paths=charts,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main(force_train: bool = False, skip_email: bool = False) -> None:
    logger.info("══════════════════════════════════════════")
    logger.info("  ElectricityForecast Pipeline")
    logger.info("  %s UTC", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("══════════════════════════════════════════")

    config.ensure_directories()
    init_db(get_engine())

    errors: list[str] = []

    # 1 — Ingest
    logger.info("═══ Step 1/5: Data Ingestion ═══")
    try:
        step_ingest()
    except DataFetchError as e:
        logger.error("Ingestion error: %s", e)
        errors.append(f"Ingestion: {e}")

    # 2 — Train
    logger.info("═══ Step 2/5: Model Training ═══")
    try:
        step_train(force=force_train)
    except (ModelError, ValueError) as e:
        logger.critical("Training failed: %s — cannot continue.", e)
        sys.exit(1)

    # 3 — Predict
    logger.info("═══ Step 3/5: Prediction ═══")
    try:
        step_predict()
    except PredictionError as e:
        logger.error("Prediction error: %s", e)
        errors.append(f"Prediction: {e}")

    # 4 — Report
    logger.info("═══ Step 4/5: Report ═══")
    report_paths: dict = {}
    try:
        report_paths = step_report()
    except ReportError as e:
        logger.error("Report error: %s", e)
        errors.append(f"Report: {e}")

    # 5 — Email
    logger.info("═══ Step 5/5: Email ═══")
    if skip_email:
        logger.info("Email skipped (--skip-email).")
    elif not report_paths:
        logger.warning("No report available — email skipped.")
    elif not config.email.is_configured():
        logger.warning("Email credentials not configured — skipping.")
    else:
        try:
            step_email(report_paths)
        except (EmailError, ConfigurationError) as e:
            logger.warning("Email failed: %s", e)
            errors.append(f"Email: {e}")

    if errors:
        logger.warning("Pipeline finished with %d error(s):", len(errors))
        for err in errors:
            logger.warning("  • %s", err)
    else:
        logger.info("Pipeline completed successfully. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Electricity forecast pipeline")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain model even if one already exists.")
    parser.add_argument("--skip-email", action="store_true",
                        help="Skip email delivery.")
    args = parser.parse_args()
    main(force_train=args.force_train, skip_email=args.skip_email)
