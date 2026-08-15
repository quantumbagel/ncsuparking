"""Entry point: run the NCSU parking data collector on a schedule."""

import logging
import signal
import sys
import time
from pathlib import Path

import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler

from accuracy import record_accuracy
from config import Config
from database import init_db, prune_predictions
from collector import poll_and_store
from events_collector import poll_and_store_events
from predict import poll_and_predict
from train import consume_train_request, retrain_if_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _wait_for_db(max_retries: int = 30, delay: int = 2) -> None:
    """Retry database connection until it succeeds or max_retries exhausted."""
    for attempt in range(1, max_retries + 1):
        try:
            init_db()
            return
        except psycopg2.OperationalError as exc:
            logger.warning(
                "DB not ready (attempt %d/%d): %s — retrying in %ds…",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to database after {max_retries} attempts")


def _mark_ready() -> None:
    Path("/tmp/collector_ready").write_text("ok")


def main() -> None:
    logger.info("Starting NCSU Parking Data Collector")
    logger.info("Poll interval: %d seconds", Config.POLL_INTERVAL)
    logger.info("Database: %s:%s/%s", Config.DB_HOST, Config.DB_PORT, Config.DB_NAME)
    logger.info("Timezone: %s  horizons: %s", Config.TIMEZONE, Config.FORECAST_HORIZONS)

    _wait_for_db()

    logger.info("Running initial parking poll...")
    poll_and_store()

    logger.info("Running initial events poll...")
    poll_and_store_events()

    logger.info("Fitting/loading baseline and first forecast...")
    retrain_if_ready()
    poll_and_predict()
    _mark_ready()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        poll_and_store, "interval",
        seconds=Config.POLL_INTERVAL, id="parking_poll", max_instances=1,
    )
    scheduler.add_job(
        poll_and_store_events, "interval",
        seconds=Config.EVENT_POLL_INTERVAL, id="events_poll", max_instances=1,
    )
    scheduler.add_job(
        poll_and_predict, "interval",
        seconds=Config.PREDICT_INTERVAL, id="prediction", max_instances=1,
    )
    scheduler.add_job(
        retrain_if_ready, "interval",
        seconds=Config.RETRAIN_CHECK_INTERVAL, id="retrain", max_instances=1,
    )
    scheduler.add_job(
        consume_train_request, "interval",
        seconds=Config.TRAIN_REQUEST_POLL, id="train_request", max_instances=1,
    )
    scheduler.add_job(
        record_accuracy, "interval",
        seconds=Config.ACCURACY_INTERVAL, id="accuracy", max_instances=1,
    )
    scheduler.add_job(
        prune_predictions, "interval",
        seconds=86400, id="retention", max_instances=1,
    )
    scheduler.start()

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
