# ⚡ ElectricityForecast

Automatisierte ML-Pipeline zur täglichen Prognose von Day-Ahead-Strompreisen in Deutschland auf Basis von [SMARD](https://www.smard.de)-Daten.

---

## Was die Pipeline macht

```
SMARD API → SQLite DB → Feature Engineering → XGBoost → 24h-Prognose → HTML-Report → Gmail
```

Täglich um **06:00 UTC** via GitHub Actions:
1. Neue Daten von SMARD holen (inkrementell)
2. ML-Modell trainieren (nur beim ersten Mal oder mit `--force-train`)
3. 24-Stunden-Prognose generieren
4. HTML + CSV Report erstellen
5. Report per Gmail versenden

---

## Schritt-für-Schritt: Projekt auf GitHub laden und starten

### Voraussetzung: Git installiert?

```bash
git --version
```

Falls nicht: [git-scm.com](https://git-scm.com/downloads) → herunterladen und installieren.

---

### 1. Projektordner entpacken

Das heruntergeladene Archiv entpacken:

```bash
tar -xzf electricity_forecast.tar.gz
cd electricity_forecast
```

Oder unter Windows: Rechtsklick auf die `.tar.gz` → "Hier entpacken".

---

### 2. Git initialisieren und ersten Commit erstellen

```bash
git init
git add .
git commit -m "feat: initial electricity forecast pipeline"
```

---

### 3. Repository auf GitHub erstellen

1. Geh auf [github.com/new](https://github.com/new)
2. Name eingeben, z. B. `electricity-forecast`
3. **Wichtig:** Repository **leer** lassen — kein README, kein .gitignore, keine Lizenz ankreuzen
4. Klick auf **"Create repository"**

---

### 4. Lokales Projekt mit GitHub verbinden

Die angezeigte URL kopieren und ausführen:

```bash
git remote add origin https://github.com/DEIN_USERNAME/electricity-forecast.git
git branch -M main
git push -u origin main
```

Remote-URL nachträglich korrigieren:
```bash
git remote set-url origin https://github.com/DEIN_USERNAME/electricity-forecast.git
```

---

### 5. GitHub Secrets setzen (für E-Mail-Versand)

Im Repository: **Settings → Secrets and variables → Actions → New repository secret**

| Secret-Name | Wert |
|---|---|
| `GMAIL_USER` | deine Gmail-Adresse, z.B. `max@gmail.com` |
| `GMAIL_PASSWORD` | 16-stelliges App-Passwort (siehe unten) |
| `EMAIL_RECEIVER` | Empfänger-Adresse |

**Gmail App-Passwort erstellen:**
1. [myaccount.google.com/security](https://myaccount.google.com/security) → 2-Faktor-Authentifizierung aktivieren
2. [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) → App-Passwort für "Mail" generieren
3. Das 16-stellige Passwort (ohne Leerzeichen) als `GMAIL_PASSWORD` eintragen

---

### 6. Pipeline manuell starten

Im Repository: **Actions → Electricity Forecast Pipeline → Run workflow**

Beim ersten Mal empfohlen:
- `force_train`: `false`
- `skip_email`: `false` (wenn Secrets gesetzt sind), sonst `true`

---

### 7. Ergebnisse prüfen

Nach dem Run:
- **Actions → letzter Run → Forecast Pipeline → Run pipeline** — hier stehen alle Logs
- **Actions → letzter Run → Artifacts** — dort liegen Report-HTML und CSV zum Download

---

## Lokale Installation (optional)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env
# .env mit Zugangsdaten befüllen

python -m src.main --skip-email  # Pipeline lokal testen
```

---

## Umgebungsvariablen

| Variable | Beschreibung | Standard |
|---|---|---|
| `GMAIL_USER` | Gmail-Absender | — |
| `GMAIL_PASSWORD` | App-Passwort | — |
| `EMAIL_RECEIVER` | Empfänger | — |
| `DATABASE_URL` | SQLAlchemy-URL | `sqlite:///data/electricity.db` |
| `MODEL_PATH` | Modell-Verzeichnis | `models/` |
| `LOG_LEVEL` | Log-Level | `INFO` |

---

## Troubleshooting

**"No new data from SMARD"**
→ Normal bei inkrementellen Updates wenn Daten aktuell sind. Kein Fehler.

**"Too little data to train"**
→ Erste Ausführung braucht ≥ 300 Stunden Daten (~13 Tage). SMARD liefert 90 Tage.

**"Gmail auth failed"**
→ Kein normales Passwort verwenden — nur App-Passwort. 2FA muss aktiv sein.

**Pipeline läuft durch aber keine E-Mail**
→ Prüfe ob alle 3 Secrets in GitHub gesetzt sind (GMAIL_USER, GMAIL_PASSWORD, EMAIL_RECEIVER).

**Tests lokal ausführen:**
```bash
pytest tests/ -v
```
