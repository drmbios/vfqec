"""RQ worker entry point; PostgreSQL/SQLite and filesystem retain results after RQ expiry."""

import os

from redis import Redis
from rq import Queue, Worker

from vfqec.config import Config
from vfqec.experiment import run_experiment
from vfqec.ledger.store import Ledger


def execute(config: dict, run_id: str) -> str:
    ledger = Ledger()
    if ledger.get(run_id)["status"] == "complete":
        return run_id
    run_experiment(Config.model_validate(config), ledger, run_id)
    return run_id


def main() -> None:
    connection = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    Worker([Queue("vfqec", connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()
