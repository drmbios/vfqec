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


def test_artifacts_reject_symlink_escape_and_untrusted_ledger_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    ledger = Ledger()
    run_id = ledger.create({})
    ledger.finish(run_id, "complete")
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir()
    private_file = tmp_path / "private.txt"
    private_file.write_text("private data")
    (run_dir / "report.pdf").symlink_to(private_file)
    client = TestClient(app)
    assert client.get(f"/runs/{run_id}/artifacts/report.pdf").status_code == 404
    assert client.get(f"/runs/{run_id}/artifacts/private.txt").status_code == 404
    (run_dir / "report.pdf").unlink()
    run_dir.rmdir()
    run_dir.symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "report.pdf").write_text("outside the run")
    assert client.get(f"/runs/{run_id}/artifacts/report.pdf").status_code == 404
    invalid_id = ledger.create({}, run_id="untrusted")
    ledger.finish(invalid_id, "complete")
    assert client.get(f"/runs/{invalid_id}/artifacts/report.pdf").status_code == 404
