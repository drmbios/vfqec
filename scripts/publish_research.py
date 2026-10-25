"""Package completed, credential-free research evidence for GitHub and Pages."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from vfqec.research import protocol_digest
from vfqec.research_analysis import write_campaign_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft", action="store_true", help="Preview; never publish draft evidence"
    )
    args = parser.parse_args()
    source = ROOT / "results" / "F1-campaign"
    target = ROOT / "benchmarks" / "F1-fourier"
    protocol = json.loads((target / "protocol.json").read_text())
    local = [json.loads(p.read_text()) for p in sorted((source / "local").glob("*.json"))]
    expected = {c["id"] for c in protocol["cases"]}
    if {r["case"]["id"] for r in local} != expected:
        raise ValueError("All frozen local cases must be present before packaging")
    for result in local:
        if result["protocol_sha256"] != protocol_digest(protocol):
            raise ValueError("A result does not match the frozen protocol")
        if result["remote_jobs"]:
            raise ValueError("Cloud reproduction must not enter local primary inference")
    summary = write_campaign_report(local, source)
    verification = {"status": "draft", "cases": [], "budget": None}
    cloud = [json.loads(p.read_text()) for p in sorted((source / "cloud").glob("*.json"))]
    if not args.draft:
        if {r["case"]["id"] for r in cloud} != set(protocol["cloud_case_ids"]):
            raise ValueError("Both predeclared cloud cross-checks must complete")
        budget = json.loads((source / "cloud-budget.json").read_text())
        if budget["halted_reason"] or budget["unsettled_jobs"] or float(budget["actual_usd"]) != 0:
            raise ValueError("Cloud budget is not fully settled at zero cost")
        ledger_ids = {job["job_id"] for result in cloud for job in result["remote_jobs"]}
        admitted_ids = {job["job_id"] for job in budget["jobs"]}
        if ledger_ids != admitted_ids or len(ledger_ids) != budget["provider_calls_admitted"]:
            raise ValueError("Cloud provenance does not reconcile with all admitted jobs")
        indexed = {r["case"]["id"]: r for r in local}
        verification.update(status="complete", budget=budget)
        for remote in cloud:
            if remote["protocol_sha256"] != protocol_digest(protocol):
                raise ValueError("Cloud result protocol mismatch")
            if not remote["remote_jobs"] or any(
                j["status"] != "complete" for j in remote["remote_jobs"]
            ):
                raise ValueError("Incomplete cloud coherent jobs")
            baseline = indexed[remote["case"]["id"]]
            comparison = {}
            for name, values in remote["methods"].items():
                reference = baseline["methods"][name]
                exact_delta = float(
                    np.max(
                        np.abs(
                            np.array(values["exact_probability"]) - reference["exact_probability"]
                        )
                    )
                )
                comparison[name] = {
                    "endpoint_counts_equal": values["errors"] == reference["errors"],
                    "cloud_final_failures": values["errors"][-1],
                    "local_final_failures": reference["errors"][-1],
                    "max_absolute_exact_probability_difference": exact_delta,
                }
            verification["cases"].append(
                {
                    "id": remote["case"]["id"],
                    "cloud_run_id": remote["run_id"],
                    "local_run_id": baseline["run_id"],
                    "backend": remote["backend"],
                    "executed_backend": "bluequbit",
                    "executed_device": "cpu",
                    "planned_case_config_backend": remote["case"]["config"]["backend"],
                    "remote_jobs": len(remote["remote_jobs"]),
                    "methods": comparison,
                    "max_control_difference": {
                        name: float(
                            np.max(np.abs(np.array(fit["theta"]) - baseline["fits"][name]["theta"]))
                        )
                        for name, fit in remote["fits"].items()
                    },
                }
            )
    summary["cloud_verification"] = {
        "status": verification["status"],
        "cases": len(verification["cases"]),
        "actual_usd": verification["budget"]["actual_usd"] if verification["budget"] else None,
        "jobs": verification["budget"]["provider_calls_admitted"] if verification["budget"] else 0,
        "authorization_usd": 10,
    }
    legacy_root = ROOT / "results" / "F1-legacy-validation"
    legacy_path = legacy_root / "suite-summary.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        if not args.draft and legacy["status"] != "complete":
            raise ValueError("Full legacy revalidation has not completed")
        (target / "legacy-summary.json").write_text(json.dumps(legacy, indent=2, allow_nan=False))
        legacy_results = [
            json.loads((legacy_root / run["result_path"]).read_text())
            for suite in legacy["suites"].values()
            for run in suite["runs"]
            if run["status"] == "complete"
        ]
        (target / "legacy-results.json.gz").write_bytes(
            gzip.compress(
                json.dumps(legacy_results, separators=(",", ":"), allow_nan=False).encode(), mtime=0
            )
        )
        summary["legacy_revalidation"] = {
            "status": legacy["status"],
            "runs": len(legacy_results),
            "quick": False,
            "total_conceptual_shots": sum(r["shots_used"] for r in legacy_results),
            "provider_jobs": 0,
            "provider_usd": 0,
        }
    elif not args.draft:
        raise ValueError("Full legacy evidence is required")
    (target / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (target / "cloud-verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False)
    )
    for name in ("comparison.png", "report.html"):
        shutil.copy2(source / name, target / name)
    for name, results in (("local-results", local), ("cloud-results", cloud)):
        payload = json.dumps(results, separators=(",", ":"), allow_nan=False).encode()
        (target / (name + ".json.gz")).write_bytes(gzip.compress(payload, mtime=0))
    manifest = {
        "status": "draft" if args.draft else "complete",
        "protocol_sha256": protocol_digest(protocol),
        "local_cases": len(local),
        "cloud_cases": len(cloud),
        "files": {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in sorted(target.iterdir())
            if p.name not in ("manifest.json", "README.md") and p.is_file()
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "local_cases": len(local),
                "cloud_cases": len(cloud),
                "cloud": summary["cloud_verification"],
            }
        )
    )


if __name__ == "__main__":
    main()
