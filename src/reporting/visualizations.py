"""Chart generation for the daily report."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def plot_historical_prices(df: pd.DataFrame, path: Path, days: int = 30) -> Path:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].dropna(subset=["price_eur_mwh"])

    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    ax.plot(df["timestamp"], df["price_eur_mwh"], lw=0.9, color="#1f77b4")
    ax.fill_between(df["timestamp"], df["price_eur_mwh"], alpha=0.1, color="#1f77b4")
    ax.set(xlabel="Zeit (UTC)", ylabel="EUR/MWh",
           title=f"Day-Ahead-Preis — letzte {days} Tage")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forecast(hist_df: pd.DataFrame, fc_df: pd.DataFrame, path: Path) -> Path:
    hist = hist_df.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    hist = hist.sort_values("timestamp").tail(72)

    fc = fc_df.copy()
    fc["forecast_timestamp"] = pd.to_datetime(fc["forecast_timestamp"], utc=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    ax.plot(hist["timestamp"], hist["price_eur_mwh"], label="Historisch", lw=1.2, color="#1f77b4")
    ax.plot(fc["forecast_timestamp"], fc["predicted_price_eur_mwh"],
            label="Prognose 24h", lw=1.5, ls="--", color="#ff7f0e")
    ax.axvline(hist["timestamp"].max(), color="grey", lw=0.8, ls=":")
    ax.set(xlabel="Zeit (UTC)", ylabel="EUR/MWh", title="Strompreis-Prognose — nächste 24h")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_load(df: pd.DataFrame, path: Path, days: int = 14) -> Path:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].dropna(subset=["load_mwh"])

    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    ax.plot(df["timestamp"], df["load_mwh"] / 1000, lw=0.9, color="#2ca02c")
    ax.set(xlabel="Zeit (UTC)", ylabel="GWh", title=f"Netzlast — letzte {days} Tage")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
