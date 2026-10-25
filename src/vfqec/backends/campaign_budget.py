"""Persistent, fail-closed admission control for a free-CPU BlueQubit campaign.

The user authorizes at most USD 10 across the entire campaign. The provider has
no per-submission hard-price-cap argument in this SDK, so this guard admits only
single CPU jobs freshly quoted at exactly USD 0, and freezes on any unknown or
nonzero actual charge. It cannot reverse a provider charge after submission.
No credentials, circuits, HTTP bodies, or exception messages are persisted.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path


class CampaignBudgetError(RuntimeError):
    """Stop the campaign without exposing provider request data or retry status."""


def _usd(value) -> Decimal:
    if value is None or isinstance(value, bool):
        raise CampaignBudgetError("Provider price must be a known nonnegative USD amount")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise CampaignBudgetError("Invalid provider USD amount") from None
    if not amount.is_finite() or amount < 0:
        raise CampaignBudgetError("Provider price must be finite and nonnegative")
    return amount


class CampaignBudgetClient:
    """BlueQubit client facade shared by every seed/run in one bounded campaign.

    Pass this object to ``BlueQubitBackend(..., client=guard)``. Reuse the same
    SQLite path across all processes. Budget and job limit are immutable once
    initialized; a restarted process cannot reset reservations. At most
    ``max_jobs`` provider ``run`` calls are admitted, including ambiguous calls.

    The underlying SDK's global HTTP retry count is set to zero when wrapping a
    real BlueQubit client: the existing backend may retry reads, but an uncertain
    submission freezes this guard and cannot reach the provider a second time.
    """

    def __init__(self, client, path, *, cap_usd="10", max_jobs=400):
        cap = _usd(cap_usd)
        if cap > Decimal("10"):
            raise CampaignBudgetError("Campaign authorization is at most USD 10")
        if not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs < 1:
            raise ValueError("max_jobs must be a positive integer")
        self.client = client
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if type(client).__module__.startswith("bluequbit."):
            from bluequbit import http_utils

            http_utils.NUMBER_OF_RETRIES = 0
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS campaign "
                "(id INTEGER PRIMARY KEY CHECK (id=1), cap_usd TEXT NOT NULL, "
                "max_jobs INTEGER NOT NULL, halted_reason TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs "
                "(reservation_id TEXT PRIMARY KEY, job_id TEXT UNIQUE, "
                "estimated_usd TEXT NOT NULL, actual_usd TEXT, status TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO campaign(id,cap_usd,max_jobs) VALUES (1,?,?)",
                (str(cap), max_jobs),
            )
            row = connection.execute("SELECT cap_usd,max_jobs FROM campaign").fetchone()
            if _usd(row[0]) != cap or row[1] != max_jobs:
                raise CampaignBudgetError("Cannot change an existing campaign's authorization")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE status IN ('reserved','uncertain') LIMIT 1"
            ).fetchone():
                # A process may have died after sending a request but before saving
                # its job ID. A new facade cannot safely assume it was never sent.
                connection.execute(
                    "UPDATE campaign SET halted_reason=COALESCE("
                    "halted_reason,'unreconciled_submission') WHERE id=1"
                )

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _halt(self, reason):
        with self._transaction() as connection:
            connection.execute(
                "UPDATE campaign SET halted_reason=COALESCE(halted_reason,?) WHERE id=1",
                (reason,),
            )

    def estimate(self, circuit, device="cpu"):
        """Return a read-only quote; run always obtains another fresh quote."""
        return self.client.estimate(circuit, device=device)

    def run(self, circuit, device="cpu", **kwargs):
        if device != "cpu" or isinstance(circuit, (list, tuple)):
            raise CampaignBudgetError("Campaign admits one CPU circuit per submission only")
        if kwargs.get("asynchronous") is not True:
            raise CampaignBudgetError("Campaign submissions must be asynchronous")
        # Check before the quote, then again inside the atomic reservation below.
        with self._transaction() as connection:
            if connection.execute("SELECT halted_reason FROM campaign").fetchone()[0]:
                raise CampaignBudgetError("Campaign is frozen; inspect its budget ledger")
        estimate = self.client.estimate(circuit, device=device)
        estimated = _usd(getattr(estimate, "estimated_cost", None))
        if estimated != 0:
            raise CampaignBudgetError("Refusing paid CPU quote; only USD 0 jobs are admitted")
        reservation = uuid.uuid4().hex
        with self._transaction() as connection:
            policy = connection.execute("SELECT * FROM campaign").fetchone()
            jobs = connection.execute("SELECT * FROM jobs").fetchall()
            if policy["halted_reason"]:
                raise CampaignBudgetError("Campaign is frozen; inspect its budget ledger")
            if len(jobs) >= policy["max_jobs"]:
                raise CampaignBudgetError("Aggregate campaign provider-job limit exhausted")
            committed = sum(
                (_usd(row["actual_usd"] or row["estimated_usd"]) for row in jobs), Decimal(0)
            )
            if committed + estimated > _usd(policy["cap_usd"]):
                raise CampaignBudgetError("Aggregate campaign USD authorization exhausted")
            connection.execute(
                "INSERT INTO jobs(reservation_id,estimated_usd,status) VALUES (?,?,'reserved')",
                (reservation, str(estimated)),
            )
        try:
            job = self.client.run(circuit, device=device, **kwargs)
            job_id = getattr(job, "job_id", None)
            if not isinstance(job_id, str) or not job_id:
                raise CampaignBudgetError("Provider returned no single job identifier")
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE jobs SET job_id=?,status='submitted' WHERE reservation_id=?",
                    (job_id, reservation),
                )
            return job
        except Exception:
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE jobs SET status='uncertain' WHERE reservation_id=?", (reservation,)
                )
                connection.execute(
                    "UPDATE campaign SET halted_reason='ambiguous_submission' WHERE id=1"
                )
            raise CampaignBudgetError(
                "Submission outcome uncertain; campaign frozen without resubmission"
            ) from None

    def wait(self, job_id, timeout=None):
        if not isinstance(job_id, str):
            raise CampaignBudgetError("Wait requires one campaign job identifier")
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise CampaignBudgetError("Job is not registered in this campaign")
        # A transient read error leaves the reservation intact and may be retried.
        result = self.client.wait(job_id, timeout=timeout)
        try:
            actual = _usd(getattr(result, "cost", None))
        except CampaignBudgetError:
            self._halt("unknown_actual_cost")
            raise CampaignBudgetError("Unknown actual charge; campaign frozen") from None
        with self._transaction() as connection:
            previous = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if previous["actual_usd"] is not None and _usd(previous["actual_usd"]) != actual:
                connection.execute(
                    "UPDATE campaign SET halted_reason='changed_actual_cost' WHERE id=1"
                )
                changed = True
            else:
                changed = False
            connection.execute(
                "UPDATE jobs SET actual_usd=?,status='settled' WHERE job_id=?",
                (str(actual), job_id),
            )
            if actual > _usd(previous["estimated_usd"]):
                connection.execute(
                    "UPDATE campaign SET halted_reason='actual_exceeded_quote' WHERE id=1"
                )
        if actual != 0 or changed:
            raise CampaignBudgetError(
                "Provider actual charge violated free-CPU quote; campaign frozen"
            )
        return result

    def summary(self):
        """Return JSON-safe totals and job IDs, without credentials or HTTP data."""
        with self._transaction() as connection:
            policy = dict(connection.execute("SELECT * FROM campaign").fetchone())
            jobs = [dict(row) for row in connection.execute("SELECT * FROM jobs ORDER BY rowid")]
        actual = sum((_usd(row["actual_usd"]) for row in jobs if row["actual_usd"] is not None))
        reserved = sum(
            (_usd(row["estimated_usd"]) for row in jobs if row["actual_usd"] is None), Decimal(0)
        )
        result = {
            "cap_usd": policy["cap_usd"],
            "max_jobs": policy["max_jobs"],
            "provider_calls_admitted": len(jobs),
            "actual_usd": str(actual),
            "unsettled_reserved_usd": str(reserved),
            "unsettled_jobs": sum(row["actual_usd"] is None for row in jobs),
            "halted_reason": policy["halted_reason"],
            "jobs": jobs,
        }
        json.dumps(result, allow_nan=False)
        return result
