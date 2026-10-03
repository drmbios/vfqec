"""Local FastAPI service backed by an RQ job queue and persistent run ledger."""

import asyncio
import json
import os
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from redis import Redis
from rq import Queue

from vfqec.config import Config
from vfqec.ledger.store import Ledger

app = FastAPI(title="VFQEC", version="0.1.0")


def queue() -> Queue:
    return Queue(
        "vfqec",
        connection=Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        default_timeout=86400,
    )


def get_run(run_id: str) -> dict:
    try:
        return Ledger().get(run_id)
    except KeyError:
        raise HTTPException(404, "Run not found") from None


@app.get("/health")
def health() -> dict:
    try:
        Ledger().list_runs(1)
        queue().connection.ping()
        return {"status": "ok"}
    except Exception:
        raise HTTPException(503, "Database or queue unavailable") from None


@app.post("/runs", status_code=202)
def start_run(config: Config) -> dict:
    if config.shot_bound > config.max_shots:
        raise HTTPException(422, f"Required shot bound is {config.shot_bound}")
    ledger = Ledger()
    run_id = ledger.create(config.model_dump())
    try:
        queue().enqueue(
            "vfqec.worker.execute",
            config.model_dump(),
            run_id,
            job_id=run_id,
            result_ttl=86400,
            failure_ttl=604800,
        )
    except Exception:
        ledger.finish(run_id, "failed", {"error": "Queue unavailable"})
        raise HTTPException(503, "Queue unavailable") from None
    return {"run_id": run_id, "status": "queued", "events": f"/runs/{run_id}/events"}


@app.get("/runs")
def list_runs() -> list[dict]:
    return Ledger().list_runs()


@app.get("/runs/{run_id}")
def run_status(run_id: str) -> dict:
    return get_run(run_id)


@app.get("/runs/{run_id}/events")
def events(run_id: str, after: int = 0) -> StreamingResponse:
    get_run(run_id)

    async def stream():
        cursor = after
        ledger = Ledger()
        while True:
            for event in ledger.progress(run_id, cursor):
                cursor = event["id"]
                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
            if ledger.get(run_id)["status"] in ("complete", "failed"):
                break
            yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}/progress")
def progress(run_id: str, after: int = 0) -> list[dict]:
    get_run(run_id)
    return Ledger().progress(run_id, after)


@app.get("/runs/{run_id}/artifacts/{filename}")
def artifact(run_id: str, filename: str) -> FileResponse:
    row = get_run(run_id)
    allowed = {
        "result.json": "result.json",
        "report.md": "report.md",
        "report.html": "report.html",
        "report.pdf": "report.pdf",
        "logical-error.png": "logical-error.png",
        "convergence.png": "convergence.png",
        "fields.png": "fields.png",
        "syndrome-circuit.qasm": "syndrome-circuit.qasm",
    }
    if filename not in allowed or row["status"] != "complete":
        raise HTTPException(404, "Artifact unavailable")
    try:
        directory = Path(os.getenv("RESULTS_DIR", "results")).resolve() / UUID(row["id"]).hex
    except ValueError:
        raise HTTPException(404, "Artifact unavailable") from None
    path = (directory / allowed[filename]).resolve()
    # Reject symlinks that escape the run, including a symlinked run directory.
    if path.parent != directory or not path.is_file():
        raise HTTPException(404, "Artifact file missing")
    return FileResponse(path)
