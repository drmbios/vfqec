import copy
import json

import pytest

from vfqec.research_analysis import analyze_campaign, write_campaign_report


def case(seed, *, group="heldout_static", cloud=False, fourier=2, reference=3):
    def method(errors):
        return {
            "errors": [errors],
            "shots": 4000,
            "logical_error": [errors / 4000],
            "exact_probability": [errors / 4000],
            "engine": "test",
            "seed": seed,
        }

    return {
        "case": {"id": f"case-{seed}", "group": group, "seed": seed, "config": {}},
        "backend": "BlueQubit coherent simulation" if cloud else "local simulation",
        "remote_jobs": [{"status": "complete"}] if cloud else [],
        "fits": {
            m: {"shots_used": s, "theta": [0], "history": [{}, {}]}
            for m, s in [("fourier", 60000), ("spsa_budget", 61000), ("spsa_reference", 604000)]
        },
        "methods": {
            "fourier": method(fourier),
            "spsa_reference": method(reference),
            "spsa_budget": method(10),
            "standard": method(20),
        },
    }


def test_complete_seed_panel_bootstraps_cases_and_excludes_cloud_copies():
    results = [case(seed) for seed in range(2001, 2025)]
    results += [case(2001, cloud=True), case(8001, group="drift")]
    summary = analyze_campaign(results)
    assert summary == analyze_campaign(list(reversed(results)))
    primary = summary["primary"]
    assert primary["complete"]
    assert primary["cases"] == 24
    assert primary["excluded_cloud_case_ids"] == ["case-2001"]
    comparison = primary["comparisons"]["fourier_vs_spsa_reference"]
    assert comparison["ci95"] == pytest.approx([-0.00025, -0.00025])
    assert comparison["superiority_supported"]
    assert comparison["noninferiority_established"]
    assert comparison["wins_ties_losses"] == {"wins": 24, "ties": 0, "losses": 0}
    assert primary["initial_shot_ratios_reference_to_fourier"]["spsa_reference"] == pytest.approx(
        604000 / 60000
    )
    assert summary["groups"]["heldout_static / cloud verification"]["cases"] == 1
    assert summary["groups"]["drift"]["cases"] == 1


def test_incomplete_panels_and_harmful_updates_do_not_claim_success():
    incomplete = analyze_campaign([case(2001)])
    comparison = incomplete["primary"]["comparisons"]["fourier_vs_spsa_reference"]
    assert comparison["ci95"] is None
    assert not comparison["conclusions_reportable"]
    assert not comparison["noninferiority_established"]
    harmful = analyze_campaign([case(seed, fourier=20, reference=0) for seed in range(2001, 2025)])
    comparison = harmful["primary"]["comparisons"]["fourier_vs_spsa_reference"]
    assert not comparison["noninferiority_established"]
    assert comparison["mean_difference"] == pytest.approx(0.005)
    assert comparison["wins_ties_losses"]["losses"] == 24


def test_duplicate_cases_and_inconsistent_readouts_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_campaign([case(2001), case(2001)])
    duplicate_seed = case(2001)
    duplicate_seed["case"]["id"] = "different-id"
    with pytest.raises(ValueError, match="distinct independent seeds"):
        analyze_campaign([case(2001), duplicate_seed])
    invalid = case(2001)
    invalid["methods"]["fourier"]["logical_error"] = [0.1]
    with pytest.raises(ValueError, match="shot accounting"):
        analyze_campaign([invalid])


def test_standalone_report_preserves_zero_results_and_escapes_case_metadata(tmp_path):
    result = case(2001, group="<script>alert(1)</script>", fourier=0, reference=0)
    original = copy.deepcopy(result)
    summary = write_campaign_report([result], tmp_path)
    assert result == original
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    page = (tmp_path / "report.html").read_text()
    assert "data:image/png;base64," in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    assert "zero observed errors" in page
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG")


def test_calibration_totals_include_later_unused_reference_updates(tmp_path):
    result = case(3401, group="drift")
    result["updates"] = {
        optimizer: [fit | {"round": round_index} for round_index in (0, 20, 40)]
        for optimizer, fit in result["fits"].items()
    }
    summary = write_campaign_report([result], tmp_path)
    costs = summary["calibration"]["spsa_reference"]
    assert costs["initial_shots"] == 604000
    assert costs["executed_shots"] == 3 * 604000
    assert costs["unused_reference_recalibration_shots"] == 2 * 604000
    assert costs["executed_probe_evaluations"] == 6
    assert costs["calibrations"] == 3
    assert summary["groups"]["drift"]["calibration"]["fourier"]["executed_shots"] == 180000
    assert "later reference updates were unused" in (tmp_path / "report.html").read_text()
