"""Budget safety contract tests; no provider requests are made."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import pytest

from vfqec.backends.campaign_budget import CampaignBudgetClient, CampaignBudgetError


class FakeClient:
    def __init__(self, quote=0, actual=0, fail_submission=False):
        self.quote = quote
        self.actual = actual
        self.fail_submission = fail_submission
        self.calls = 0
        self.quotes = 0
        self.lock = Lock()

    def estimate(self, circuit, device="cpu"):
        self.quotes += 1
        return SimpleNamespace(estimated_cost=self.quote)

    def run(self, circuit, **kwargs):
        with self.lock:
            self.calls += 1
            job_id = f"job-{self.calls}"
        if self.fail_submission:
            raise ConnectionError("private provider response must not be persisted")
        return SimpleNamespace(job_id=job_id)

    def wait(self, job_id, timeout=None):
        return SimpleNamespace(job_id=job_id, cost=self.actual, ok=True)


@pytest.mark.parametrize("quote", [None, -1, float("nan"), float("inf"), 0.01, True])
def test_unknown_invalid_or_paid_quotes_never_submit(tmp_path, quote):
    client = FakeClient(quote=quote)
    guard = CampaignBudgetClient(client, tmp_path / "campaign.db")
    with pytest.raises(CampaignBudgetError):
        guard.run(object(), asynchronous=True)
    assert client.calls == 0
    assert guard.summary()["provider_calls_admitted"] == 0


def test_fresh_quote_is_checked_instead_of_an_earlier_free_estimate(tmp_path):
    client = FakeClient()
    guard = CampaignBudgetClient(client, tmp_path / "campaign.db")
    assert guard.estimate(object()).estimated_cost == 0
    client.quote = 0.02
    with pytest.raises(CampaignBudgetError, match="paid CPU quote"):
        guard.run(object(), asynchronous=True)
    assert client.quotes == 2
    assert client.calls == 0


def test_zero_cost_jobs_settle_and_shared_limit_survives_restarts(tmp_path):
    client = FakeClient()
    path = tmp_path / "campaign.db"
    guard = CampaignBudgetClient(client, path, max_jobs=2)
    first = guard.run(object(), asynchronous=True)
    assert guard.wait(first.job_id).ok
    resumed = CampaignBudgetClient(client, path, max_jobs=2)
    second = resumed.run(object(), asynchronous=True)
    assert resumed.wait(second.job_id).ok
    with pytest.raises(CampaignBudgetError, match="job limit"):
        guard.run(object(), asynchronous=True)
    assert client.calls == 2
    assert resumed.summary()["unsettled_jobs"] == 0
    assert resumed.summary()["actual_usd"] == "0"


@pytest.mark.parametrize("actual", [None, -0.01, float("nan"), float("inf"), 0.01])
def test_unknown_or_changed_actual_charge_freezes_all_future_submissions(tmp_path, actual):
    client = FakeClient(actual=actual)
    path = tmp_path / "campaign.db"
    guard = CampaignBudgetClient(client, path)
    job = guard.run(object(), asynchronous=True)
    with pytest.raises(CampaignBudgetError, match="frozen"):
        guard.wait(job.job_id)
    resumed = CampaignBudgetClient(client, path)
    with pytest.raises(CampaignBudgetError, match="frozen"):
        resumed.run(object(), asynchronous=True)
    assert client.calls == 1
    assert resumed.summary()["halted_reason"] is not None
    if actual == 0.01:
        assert resumed.summary()["actual_usd"] == "0.01"


def test_ambiguous_submission_is_reserved_and_never_resubmitted(tmp_path):
    client = FakeClient(fail_submission=True)
    path = tmp_path / "campaign.db"
    guard = CampaignBudgetClient(client, path)
    with pytest.raises(CampaignBudgetError, match="uncertain"):
        guard.run(object(), asynchronous=True)
    with pytest.raises(CampaignBudgetError, match="frozen"):
        CampaignBudgetClient(client, path).run(object(), asynchronous=True)
    assert client.calls == 1
    summary = guard.summary()
    assert summary["provider_calls_admitted"] == summary["unsettled_jobs"] == 1
    assert "private provider response" not in str(summary)


def test_concurrent_clients_cannot_oversubscribe_job_reservations(tmp_path):
    client = FakeClient()
    path = tmp_path / "campaign.db"
    guards = [CampaignBudgetClient(client, path, max_jobs=3) for _ in range(8)]

    def submit(guard):
        try:
            return guard.run(object(), asynchronous=True).job_id
        except CampaignBudgetError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(submit, guards))
    assert len([value for value in accepted if value]) == 3
    assert client.calls == guards[0].summary()["provider_calls_admitted"] == 3


def test_campaign_policy_cannot_be_reset_or_increased(tmp_path):
    path = tmp_path / "campaign.db"
    CampaignBudgetClient(FakeClient(), path)
    with pytest.raises(CampaignBudgetError, match="Cannot change"):
        CampaignBudgetClient(FakeClient(), path, max_jobs=401)
    with pytest.raises(CampaignBudgetError, match="at most USD 10"):
        CampaignBudgetClient(FakeClient(), tmp_path / "other.db", cap_usd="10.01")


def test_paid_devices_batches_sync_and_foreign_waits_are_rejected(tmp_path):
    client = FakeClient()
    guard = CampaignBudgetClient(client, tmp_path / "campaign.db")
    for circuit, kwargs in [
        (object(), {"device": "gpu", "asynchronous": True}),
        ([object()], {"asynchronous": True}),
        (object(), {}),
    ]:
        with pytest.raises(CampaignBudgetError):
            guard.run(circuit, **kwargs)
    with pytest.raises(CampaignBudgetError, match="not registered"):
        guard.wait("unknown-job")
    assert client.calls == 0


def test_sdk_retry_loop_is_disabled_for_real_sdk_client(tmp_path, monkeypatch):
    from bluequbit import http_utils

    monkeypatch.setattr(http_utils, "NUMBER_OF_RETRIES", 5)
    sdk_contract_client = type("SDKContractClient", (), {"__module__": "bluequbit.test"})()
    CampaignBudgetClient(sdk_contract_client, tmp_path / "campaign.db")
    assert http_utils.NUMBER_OF_RETRIES == 0


def test_process_crash_reservation_cannot_be_ignored_on_restart(tmp_path):
    path = tmp_path / "campaign.db"
    client = FakeClient()
    CampaignBudgetClient(client, path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO jobs(reservation_id,estimated_usd,status) VALUES ('crash','0','reserved')"
        )
    resumed = CampaignBudgetClient(client, path)
    with pytest.raises(CampaignBudgetError, match="frozen"):
        resumed.run(object(), asynchronous=True)
    assert client.calls == 0
