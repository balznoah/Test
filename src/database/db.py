"""Database engine, session factory, and schema initialisation."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base
from src.utils.config import config
from src.utils.exceptions import DatabaseError
from src.utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = config.database.url
        logger.info("Connecting to database: %s", url)
        _engine = create_engine(url, echo=config.database.echo, pool_pre_ping=True)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def set_pragmas(conn, _):
                c = conn.cursor()
                c.execute("PRAGMA journal_mode=WAL;")
                c.execute("PRAGMA foreign_keys=ON;")
                c.close()
        Base.metadata.create_all(bind=_engine)
        logger.info("Database schema ready.")
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Ensure all tables exist."""
    e = engine or get_engine()
    Base.metadata.create_all(bind=e)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        factory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    else:
        factory = _SessionLocal

    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        raise DatabaseError(f"Session error: {exc}") from exc
    finally:
        session.close()
