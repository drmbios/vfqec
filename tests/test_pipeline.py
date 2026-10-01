import json

import pytest
from fastapi.testclient import TestClient

from vfqec.agents.orchestrator import PlannerAgent
from vfqec.api.app import app
from vfqec.config import Config, preset
from vfqec.experiment import METHODS, run_experiment
from vfqec.ledger.store import Ledger


def test_pipeline_reports_ledger_and_all_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    config = preset("E1", shots=64, evaluation_shots=128, iterations=2, rounds=4, cadence=2)
    result = run_experiment(config)
    assert set(result["methods"]) == set(METHODS)
    assert result["shots_used"] == config.shot_bound
    row = Ledger().get(result["run_id"])
    assert row["status"] == "complete"
    for filename in ("report.pdf", "report.html", "report.md", "logical-error.png", "result.json"):
        assert (tmp_path / result["run_id"] / filename).stat().st_size > 100
    assert len(result["true_fields"]) == 4
    client = TestClient(app)
    assert client.get(f"/runs/{result['run_id']}").status_code == 200
    assert (
        client.get(f"/runs/{result['run_id']}/events")
        .headers["content-type"]
        .startswith("text/event-stream")
    )
    assert client.get(f"/runs/{result['run_id']}/artifacts/report.pdf").status_code == 200
    assert client.get("/runs/not-a-run").status_code == 404
    assert all(
        "eps" not in json.dumps(e["payload"])
        for e in Ledger().progress(result["run_id"])
        if e["payload"]["type"] == "optimizer"
    )


def test_budget_and_invalid_physics(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="budget"):
        run_experiment(preset("E1", max_shots=1))
    assert Ledger().list_runs()[0]["status"] == "failed"
    with pytest.raises(ValueError, match="protected axis"):
        Config(eps_z=[0.2])
    with pytest.raises(ValueError, match="surface d=3"):
        Config(code="surface", distance=5, eps_x=[0])


def test_planner_respects_total_budget_and_prunes():
    plan = PlannerAgent().plan(
        "Compare repetition distances",
        20_000_000,
        [{"p": 0.003, "distance": 3, "logical_error": 0.48}],
    )
    assert plan["shot_bound"] <= plan["max_shots"]
    assert any(row["reason"] == "saturated" for row in plan["pruned"])
    assert all(c["p"] != 0.003 for c in plan["runs"])
