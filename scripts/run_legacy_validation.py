"""Re-run full legacy E1-E4 presets locally, preserving resumable suite evidence.

This driver intentionally does not load .env, initialize BlueQubit, or submit
hardware work. E3 and E4 each retain the CLI's separate 20-million-shot sweep
cap. The complete group of suites is not represented as a single capped sweep.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/F1-legacy-validation"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["RESULTS_DIR"] = str(root)
    os.environ["DATABASE_URL"] = f"sqlite:///{root / 'ledger.db'}"
    os.environ["LLM_PROVIDER"] = "none"
    os.environ["BACKEND"] = "local"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    from vfqec.config import preset
    from vfqec.experiment import run_experiment
    from vfqec.report.analysis import analyze_sweep

    groups = {
        "E1": [preset("E1", backend="local")],
        "E2-Z": [preset("E2", backend="local", basis="Z")],
        "E2-X": [preset("E2", backend="local", basis="X")],
        "E3": [
            preset("E3", backend="local", distance=d, p=p, eps_x=[0.12])
            for d in (3, 5, 7)
            for p in (0.001, 0.005, 0.02)
        ],
        "E4": [
            preset("E4", backend="local", cadence=k, shots=s)
            for k in (5, 15, 30)
            for s in (500, 2000)
        ],
    }
    summary_path = root / "suite-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "kind": "full local legacy revalidation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": "local simulation / NumPy",
            "quick": False,
            "provider_jobs": 0,
            "provider_usd": 0,
            "suite_shot_cap": 20_000_000,
            "E5": {
                "submitted": False,
                "reason": "Disabled hardware template; no hardware validation claimed",
            },
            "suites": {},
        }

    def save():
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False))
        tmp.replace(summary_path)

    started = time.monotonic()
    for name, configs in groups.items():
        bound = sum(config.shot_bound for config in configs)
        group = summary["suites"].setdefault(
            name,
            {"shot_bound": bound, "expected_runs": len(configs), "runs": [], "status": "pending"},
        )
        if bound > summary["suite_shot_cap"]:
            group.update(status="refused", reason="Full preset sweep exceeds 20-million shot cap")
            save()
            continue
        group["status"] = "running"
        save()
        for index, config in enumerate(configs):
            previous = next((row for row in group["runs"] if row["index"] == index), None)
            if previous is not None and previous["config"] != config.model_dump():
                raise ValueError("Resume configuration differs; use a separate output directory")
            if previous is not None and previous["status"] == "complete":
                continue
            print(json.dumps({"suite": name, "index": index, "status": "started"}), flush=True)
            run_started = time.monotonic()
            try:
                result = run_experiment(config)
            except Exception as exc:
                row = {
                    "index": index,
                    "status": "failed",
                    "config": config.model_dump(),
                    "error_type": type(exc).__name__,
                    "seconds": round(time.monotonic() - run_started, 3),
                }
            else:
                row = {
                    "index": index,
                    "status": "complete",
                    "config": config.model_dump(),
                    "run_id": result["run_id"],
                    "commit": result["commit"],
                    "shots_used": result["shots_used"],
                    "remote_jobs": len(result["remote_jobs"]),
                    "seconds": round(time.monotonic() - run_started, 3),
                    "result_path": f"{result['run_id']}/result.json",
                    "final_methods": {
                        method: {
                            "errors": data["errors"][-1],
                            "shots": data["shots"],
                            "logical_error": data["logical_error"][-1],
                        }
                        for method, data in result["methods"].items()
                    },
                    "analysis": result["analysis"],
                }
            group["runs"] = [old for old in group["runs"] if old["index"] != index] + [row]
            group["runs"].sort(key=lambda item: item["index"])
            save()
            print(json.dumps({"suite": name, "index": index, **row}), flush=True)
        group["status"] = (
            "complete"
            if len(group["runs"]) == len(configs)
            and all(row["status"] == "complete" for row in group["runs"])
            else "failed"
        )
        if name in ("E3", "E4") and group["status"] == "complete":
            results = [json.loads((root / row["result_path"]).read_text()) for row in group["runs"]]
            group["sweep_analysis"] = analyze_sweep(results)
        save()
    summary["status"] = (
        "complete"
        if all(group["status"] == "complete" for group in summary["suites"].values())
        else "incomplete"
    )
    summary["last_invocation_seconds"] = round(time.monotonic() - started, 3)
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
