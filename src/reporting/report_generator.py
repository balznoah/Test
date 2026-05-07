"""Generates CSV + HTML daily reports with base64-embedded images."""

import base64
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import BaseLoader, Environment, select_autoescape

from src.database.repository import ElectricityRepository, PredictionRepository
from src.ml.model_manager import ModelManager
from src.reporting.visualizations import plot_forecast, plot_historical_prices, plot_load
from src.utils.config import config
from src.utils.exceptions import ReportError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Strompreis-Prognose</title>
<style>
body{font-family:Arial,sans-serif;background:#f5f7fa;color:#333;margin:0;padding:0}
.wrap{max-width:960px;margin:0 auto;padding:24px}
header{background:#1a237e;color:#fff;padding:20px;border-radius:8px 8px 0 0}
header h1{margin:0;font-size:1.5rem}
header p{margin:4px 0 0;opacity:.8;font-size:.85rem}
.card{background:#fff;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);margin:14px 0;padding:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.box{background:#e8eaf6;border-radius:6px;padding:12px;text-align:center}
.box .val{font-size:1.8rem;font-weight:700;color:#1a237e}
.box .lbl{font-size:.75rem;color:#555;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th{background:#1a237e;color:#fff;padding:7px 10px;text-align:left}
td{padding:6px 10px;border-bottom:1px solid #eee}
tr:hover td{background:#f0f4ff}
img{max-width:100%;border-radius:6px;margin:6px 0}
footer{text-align:center;color:#aaa;font-size:.78rem;padding:14px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>⚡ Strompreis-Prognose — Deutschland</h1>
  <p>Erstellt: {{ ts }} UTC &nbsp;|&nbsp; Modell: {{ version }}</p>
</header>

<div class="card">
  <h2>Modellgüte (Cross-Validation)</h2>
  <div class="grid">
    <div class="box"><div class="val">{{ "%.2f"|format(mae) }}</div><div class="lbl">MAE EUR/MWh</div></div>
    <div class="box"><div class="val">{{ "%.2f"|format(rmse) }}</div><div class="lbl">RMSE EUR/MWh</div></div>
    <div class="box"><div class="val">{{ "%.1f"|format(mape) }}%</div><div class="lbl">MAPE</div></div>
    <div class="box"><div class="val">{{ train_rows }}</div><div class="lbl">Trainingszeilen</div></div>
  </div>
</div>

<div class="card">
  <h2>24-Stunden-Prognose</h2>
  <p>Ø <strong>{{ "%.2f"|format(avg_fc) }} EUR/MWh</strong>
     &nbsp;|&nbsp; Min {{ "%.2f"|format(min_fc) }}
     &nbsp;|&nbsp; Max {{ "%.2f"|format(max_fc) }}</p>
  <img src="{{ fc_chart }}" alt="Prognose">
  <table>
    <tr><th>Zeitstempel (UTC)</th><th>Preis (EUR/MWh)</th></tr>
    {% for r in rows %}
    <tr><td>{{ r.ts }}</td><td>{{ "%.2f"|format(r.price) }}</td></tr>
    {% endfor %}
  </table>
</div>

<div class="card">
  <h2>Historische Preise</h2>
  <img src="{{ hist_chart }}" alt="Historisch">
</div>

<div class="card">
  <h2>Netzlast</h2>
  <img src="{{ load_chart }}" alt="Netzlast">
</div>

<footer>ElectricityForecast &nbsp;|&nbsp; Daten: SMARD / Bundesnetzagentur</footer>
</div>
</body>
</html>"""


class ReportGenerator:
    def __init__(self) -> None:
        self._reports_dir = config.data.reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._elec = ElectricityRepository()
        self._pred = PredictionRepository()
        self._manager = ModelManager()

    def generate(self) -> dict[str, Path]:
        version = self._manager.get_latest_version()
        if not version:
            raise ReportError("No model version found. Train the model first.")

        meta = self._manager.load_metadata()
        fc_df = self._pred.get_latest_predictions(model_version=version)
        if fc_df.empty:
            raise ReportError("No predictions found. Run prediction step first.")

        hist_df = self._elec.get_all_records()
        ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Charts
        hist_chart = plot_historical_prices(hist_df, self._reports_dir / f"hist_{ts_str}.png")
        fc_chart = plot_forecast(hist_df, fc_df, self._reports_dir / f"fc_{ts_str}.png")
        load_chart = plot_load(hist_df, self._reports_dir / f"load_{ts_str}.png")

        # CSV
        csv_path = self._reports_dir / f"forecast_{ts_str}.csv"
        fc_df.to_csv(csv_path, index=False)

        # Embed images as base64 so they display correctly in email clients
        def _b64(path: Path) -> str:
            return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

        # HTML
        rows = []
        for _, r in fc_df.iterrows():
            ts = r["forecast_timestamp"]
            rows.append({"ts": ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
                         "price": float(r["predicted_price_eur_mwh"])})

        env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
        html = env.from_string(_HTML).render(
            ts=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            version=version,
            mae=meta.get("cv_mae", 0),
            rmse=meta.get("cv_rmse", 0),
            mape=meta.get("cv_mape", 0),
            train_rows=meta.get("train_rows", 0),
            avg_fc=fc_df["predicted_price_eur_mwh"].mean(),
            min_fc=fc_df["predicted_price_eur_mwh"].min(),
            max_fc=fc_df["predicted_price_eur_mwh"].max(),
            rows=rows,
            fc_chart=_b64(fc_chart),
            hist_chart=_b64(hist_chart),
            load_chart=_b64(load_chart),
        )
        html_path = self._reports_dir / f"report_{ts_str}.html"
        html_path.write_text(html, encoding="utf-8")

        logger.info("Report generated: %s", html_path)
        return {"csv": csv_path, "html": html_path,
                "fc_chart": fc_chart, "hist_chart": hist_chart, "load_chart": load_chart}
