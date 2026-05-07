"""Gmail SMTP email delivery."""

import smtplib
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.utils.config import config
from src.utils.exceptions import ConfigurationError, EmailError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)


class EmailSender:
    def __init__(self) -> None:
        self._cfg = config.email

    def send_daily_report(
        self,
        html_path: Path,
        csv_path: Path,
        chart_paths: list[Path] | None = None,
    ) -> None:
        if not self._cfg.is_configured():
            raise ConfigurationError(
                "Email credentials not set. Check GMAIL_USER, GMAIL_PASSWORD, EMAIL_RECEIVER."
            )

        date_str = datetime.now(tz=timezone.utc).strftime("%d.%m.%Y")
        subject = f"⚡ Strompreis-Prognose {date_str}"
        body = html_path.read_text(encoding="utf-8")

        msg = MIMEMultipart("mixed")
        msg["From"] = self._cfg.gmail_user
        msg["To"] = self._cfg.email_receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html", "utf-8"))

        for p in [html_path, csv_path] + (chart_paths or []):
            try:
                self._attach(msg, p)
            except OSError as e:
                logger.warning("Could not attach %s: %s", p, e)

        try:
            with smtplib.SMTP(self._cfg.smtp_host, self._cfg.smtp_port, timeout=30) as srv:
                srv.ehlo()
                srv.starttls()
                srv.ehlo()
                srv.login(self._cfg.gmail_user, self._cfg.gmail_password)
                srv.sendmail(self._cfg.gmail_user, self._cfg.email_receiver, msg.as_string())
            logger.info("Email sent to %s.", self._cfg.email_receiver)
        except smtplib.SMTPAuthenticationError as e:
            raise EmailError(
                "Gmail auth failed. Use an App Password, not your account password."
            ) from e
        except smtplib.SMTPException as e:
            raise EmailError(f"SMTP error: {e}") from e
        except OSError as e:
            raise EmailError(f"Network error: {e}") from e

    def _attach(self, msg: MIMEMultipart, path: Path) -> None:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=path.name)
        msg.attach(part)
