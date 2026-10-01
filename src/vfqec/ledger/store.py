"""Transactional SQLite/Postgres ledger; no API keys are ever persisted."""

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError


class ExperimentLedger:
    """Persistent provenance for runs, optimizer observations, and accepted provider jobs."""

    def __init__(self, url: str | None = None):
        root = Path(os.getenv("RESULTS_DIR", "results"))
        root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            url or os.getenv("DATABASE_URL", f"sqlite:///{root}/ledger.db"), pool_pre_ping=True
        )
        meta = MetaData()
        self.runs = Table(
            "runs",
            meta,
            Column("id", String, primary_key=True),
            Column("created", String),
            Column("status", String),
            Column("config", Text),
            Column("result", Text),
            Column("commit", String),
        )
        self.events = Table(
            "events",
            meta,
            Column("id", Integer, primary_key=True),
            Column("run_id", String, index=True),
            Column("created", String),
            Column("payload", Text),
        )
        self.jobs = Table(
            "jobs",
            meta,
            Column("key", String, primary_key=True),
            Column("run_id", String),
            Column("backend", String),
            Column("job_id", String),
            Column("status", String),
            Column("cost", Float),
            Column("payload", Text),
        )
        meta.create_all(self.engine)

    def create(self, config: dict, run_id: str | None = None) -> str:
        run_id = run_id or uuid.uuid4().hex
        commit = os.getenv("VFQEC_GIT_COMMIT", "unversioned")
        if commit == "unversioned":
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
                ).strip()
                dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
                if dirty:
                    commit += "+dirty"
            except (OSError, subprocess.CalledProcessError):
                commit = "unversioned"
        with self.engine.begin() as conn:
            conn.execute(
                insert(self.runs).values(
                    id=run_id,
                    created=self.now(),
                    status="queued",
                    config=json.dumps(config),
                    result=None,
                    commit=commit,
                )
            )
        return run_id

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def event(self, run_id: str, payload: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(self.events).values(
                    run_id=run_id, created=self.now(), payload=json.dumps(payload)
                )
            )

    def finish(self, run_id: str, status: str, result: dict | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(self.runs)
                .where(self.runs.c.id == run_id)
                .values(status=status, result=json.dumps(result) if result is not None else None)
            )

    def get(self, run_id: str) -> dict:
        with self.engine.connect() as conn:
            row = conn.execute(select(self.runs).where(self.runs.c.id == run_id)).mappings().first()
        if row is None:
            raise KeyError(run_id)
        result = dict(row)
        for key in ("config", "result"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self.runs.c.id, self.runs.c.created, self.runs.c.status, self.runs.c.commit)
                .order_by(self.runs.c.created.desc())
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def progress(self, run_id: str, after: int = 0) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self.events)
                .where(self.events.c.run_id == run_id, self.events.c.id > after)
                .order_by(self.events.c.id)
            ).mappings()
            return [dict(row, payload=json.loads(row["payload"])) for row in rows]

    def claim_job(self, key: str, run_id: str, backend: str) -> dict | None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(self.jobs).values(
                        key=key, run_id=run_id, backend=backend, status="claimed", cost=0.0
                    )
                )
            return None
        except IntegrityError:
            with self.engine.connect() as conn:
                row = conn.execute(select(self.jobs).where(self.jobs.c.key == key)).mappings().one()
                return dict(row)

    def save_job(self, key: str, **values) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(self.jobs).where(self.jobs.c.key == key).values(**values))

    def run_jobs(self, run_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    select(self.jobs).where(self.jobs.c.run_id == run_id)
                ).mappings()
            ]


Ledger = ExperimentLedger
