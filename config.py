"""Application configuration loaded from environment variables."""

import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def _csv_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = (raw or "").strip()
    if not text:
        return default
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


class Config:
    """Central configuration loaded from environment with sensible defaults."""

    # PostgreSQL
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "ncsuparking")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # Display + feature clock. Snapshots stay UTC in the database.
    TIMEZONE: str = os.getenv("TIMEZONE", "America/New_York")

    # Polling
    POLL_INTERVAL: int = int(os.getenv("POLL_INTERVAL", "10"))
    EVENT_POLL_INTERVAL: int = int(os.getenv("EVENT_POLL_INTERVAL", "3600"))
    PREDICT_INTERVAL: int = int(os.getenv("PREDICT_INTERVAL", "60"))

    # Forecasting
    GRID_MINUTES: int = int(os.getenv("GRID_MINUTES", "5"))
    FORECAST_HORIZONS: tuple[int, ...] = _csv_ints(
        os.getenv("FORECAST_HORIZONS", ""),
        (15, 30, 60, 120, 180, 360, 720, 1440),
    )
    FORECAST_MINUTES: int = int(os.getenv("FORECAST_MINUTES", "1440"))
    RETRAIN_INTERVAL: int = int(os.getenv("RETRAIN_INTERVAL", "86400"))
    RETRAIN_CHECK_INTERVAL: int = int(os.getenv("RETRAIN_CHECK_INTERVAL", "3600"))
    TRAIN_REQUEST_POLL: int = int(os.getenv("TRAIN_REQUEST_POLL", "15"))
    ACCURACY_INTERVAL: int = int(os.getenv("ACCURACY_INTERVAL", "3600"))
    PREDICTION_RETENTION_DAYS: int = int(os.getenv("PREDICTION_RETENTION_DAYS", "14"))
    MIN_TRAIN_ROWS: int = int(os.getenv("MIN_TRAIN_ROWS", "500"))

    # APIs
    PARKING_API_URL: str = os.getenv(
        "PARKING_API_URL",
        "https://transportation.ncsu.edu/wp-json/ncsu-transportation-parking-view/v1/get-parking-data",
    )
    EVENTS_API_URL: str = os.getenv(
        "EVENTS_API_URL",
        "https://calendar.ncsu.edu/api/2/events",
    )
    EVENT_DAYS: int = int(os.getenv("EVENT_DAYS", "30"))

    @classmethod
    def zoneinfo(cls) -> ZoneInfo:
        return ZoneInfo(cls.TIMEZONE)

    @classmethod
    def db_conn_string(cls) -> str:
        return (
            f"host={cls.DB_HOST} port={cls.DB_PORT} "
            f"dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD}"
        )
