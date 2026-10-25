import json
from collections import Counter

import numpy as np
import pytest

import vfqec.research as research
from vfqec.config import Config
from vfqec.ledger.store import Ledger


def test_frozen_protocol_has_disjoint_heldout_seeds_and_declared_extensions():
    protocol = research.make_protocol()
    assert protocol == research.make_protocol()
    assert len(protocol["cases"]) == 48
    groups = Counter(case["group"] for case in protocol["cases"])
    assert groups["development"] == 4
    assert groups["heldout_static"] == 24
    assert len({case["id"] for case in protocol["cases"]}) == 48
    assert len({case["seed"] for case in protocol["cases"]}) == 48
    assert all(case["config"]["window"] == 1 for case in protocol["cases"])
    assert set(protocol["cloud_case_ids"]) <= {case["id"] for case in protocol["cases"]}
    development = {case["seed"] for case in protocol["cases"] if case["group"] == "development"}
    heldout = {case["seed"] for case in protocol["cases"] if case["group"] == "heldout_static"}
    assert development.isdisjoint(heldout)
    static = [case for case in protocol["cases"] if case["group"] == "heldout_static"]
    for case in static:
        fields = np.array(case["config"]["eps_x"])
        assert len(fields) == 3
        assert np.all(np.abs(fields) <= 0.35)
        assert case["config"]["rounds"] == 60
        assert case["config"]["shots"] == 1000
        assert case["config"]["iterations"] == 20
    assert not np.array_equal(static[0]["config"]["eps_x"], static[1]["config"]["eps_x"])
    surfaces = [case["config"] for case in protocol["cases"] if "surface" in case["group"]]
    assert {(case["basis"], case["zz"]) for case in surfaces} == {
        ("X", 0.0),
        ("X", 0.08),
        ("Z", 0.0),
        ("Z", 0.08),
    }
    assert all(case["code"] == "surface" and case["distance"] == 3 for case in surfaces)
    for case in protocol["cases"]:
        if case["group"] == "phase_memory":
            assert case["config"]["basis"] == "X"
            assert case["config"]["eps_x"] == [0.0]
            assert len(case["config"]["eps_z"]) == 3


def tiny_case(group="heldout_static", **changes):
    config = Config(
        experiment="F1",
        window=1,
        rounds=2,
        shots=32,
        evaluation_shots=64,
        iterations=2,
        cadence=2,
        seed=47,
        max_shots=2_000_000,
    ).model_copy(update=changes)
    return {"id": "test-case", "group": group, "seed": 47, "config": config.model_dump()}


def assert_complete_shot_accounting(result):
    calibration = sum(
        update["shots_used"] for updates in result["updates"].values() for update in updates
    )
    for updates in result["updates"].values():
        for update in updates:
            assert sum(row["shots"] for row in update["history"]) == update["shots_used"]
    endpoints = sum(
        len(method["errors"]) * method["shots"] for method in result["methods"].values()
    )
    assert result["shots_used"] == calibration + endpoints


def test_real_case_preserves_optimizer_observation_boundary_and_total_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    state = {"fitting": False, "observations": 0}
    oracle_truth = research.HiddenPlant.oracle_theta

    def guarded_truth(self, round_index):
        assert not state["fitting"], "Truth leaked into optimizer fitting"
        return oracle_truth(self, round_index)

    monkeypatch.setattr(research.HiddenPlant, "oracle_theta", guarded_truth)
    for optimizer in (research.FieldOptimizer, research.FourierOptimizer):
        original_fit = optimizer.fit

        def fit_with_boundary(self, oracle, initial=None, original_fit=original_fit):
            assert callable(oracle)

            def observations_only(point, shots, seed):
                observation = oracle(point, shots, seed)
                assert set(vars(observation)) == {"cost", "shots", "checks"}
                assert observation.shots == shots
                assert 0 <= observation.cost <= 1
                state["observations"] += 1
                return observation

            state["fitting"] = True
            try:
                return original_fit(self, observations_only, initial)
            finally:
                state["fitting"] = False

        monkeypatch.setattr(optimizer, "fit", fit_with_boundary)
    result = research.run_case(tiny_case(), research.make_protocol())
    assert state["observations"] > 0
    assert result["fits"]["spsa_budget"]["shots_used"] == 224
    assert result["fits"]["fourier"]["shots_used"] == 160
    assert result["fits"]["spsa_reference"]["shots_used"] == 604_000
    assert len(result["methods"]) == 7
    assert result["remote_jobs"] == []
    assert result["protocol_sha256"] == research.protocol_digest(research.make_protocol())
    assert Ledger().get(result["run_id"])["status"] == "complete"
    saved = json.loads((tmp_path / result["run_id"] / "result.json").read_text())
    assert saved["shots_used"] == result["shots_used"]
    assert_complete_shot_accounting(result)


def test_drift_accounting_includes_every_recalibration_and_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    result = research.run_case(
        tiny_case("drift", rounds=3, cadence=1, evaluation_shots=16, drift_amplitude=0.15),
        research.make_protocol(),
    )
    for name in ("spsa_budget", "fourier"):
        assert [update["round"] for update in result["updates"][name]] == [0, 1, 2]
    assert {"adaptive_spsa", "adaptive_fourier"} <= set(result["methods"])
    assert result["shots_used"] <= result["case"]["config"]["max_shots"]
    assert_complete_shot_accounting(result)


def test_research_case_stops_at_shot_cap_and_persists_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with pytest.raises(RuntimeError, match="shot budget"):
        research.run_case(tiny_case(max_shots=224), research.make_protocol())
    runs = Ledger().list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "failed"
    failure = Ledger().get(runs[0]["id"])["result"]
    assert failure["error_type"] == "RuntimeError"
    assert not (tmp_path / runs[0]["id"] / "result.json").exists()
