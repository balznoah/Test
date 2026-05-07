"""SQLAlchemy ORM models."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ElectricityRecord(Base):
    __tablename__ = "electricity_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price_eur_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    load_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("timestamp", name="uq_electricity_timestamp"),
        Index("ix_electricity_ts", "timestamp"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_price_eur_mwh: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("forecast_timestamp", "model_version", name="uq_pred_ts_version"),
        Index("ix_pred_fc_ts", "forecast_timestamp"),
    )


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
