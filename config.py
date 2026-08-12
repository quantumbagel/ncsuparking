"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration loaded from environment with sensible defaults."""

    # PostgreSQL
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "ncsuparking")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # Polling
    POLL_INTERVAL: int = int(os.getenv("POLL_INTERVAL", "10"))
    EVENT_POLL_INTERVAL: int = int(os.getenv("EVENT_POLL_INTERVAL", "3600"))
    PREDICT_INTERVAL: int = int(os.getenv("PREDICT_INTERVAL", "60"))

    # Forecasting
    FORECAST_MINUTES: int = int(os.getenv("FORECAST_MINUTES", "1440"))  # 24h at 1-min granularity
    RETRAIN_INTERVAL: int = int(os.getenv("RETRAIN_INTERVAL", "86400"))  # daily
    ACCURACY_INTERVAL: int = int(os.getenv("ACCURACY_INTERVAL", "3600"))  # hourly

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
    def db_conn_string(cls) -> str:
        return (
            f"host={cls.DB_HOST} port={cls.DB_PORT} "
            f"dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD}"
        )
