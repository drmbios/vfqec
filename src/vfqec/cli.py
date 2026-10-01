"""Command-line interface, sweep expansion, and disabled-by-default hardware template."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from vfqec.agents.orchestrator import PlannerAgent
from vfqec.config import Config, preset
from vfqec.experiment import run_experiment
from vfqec.ledger.store import Ledger
from vfqec.report.analysis import analyze_sweep
from vfqec.report.generate import generate_report, sweep_figure


def run_sweep(plan: dict) -> list[dict]:
    configs = [Config.model_validate(row) for row in plan["runs"]]
    if len(configs) > 100:
        raise ValueError("At most 100 runs per sweep")
    if sum(c.shot_bound for c in configs) > plan.get("max_shots", 20_000_000):
        raise ValueError("Aggregate sweep shot budget exceeded")
    if sum(c.max_credits for c in configs if c.backend == "bluequbit") > plan.get(
        "max_credits", 10
    ):
        raise ValueError("Aggregate sweep credit allocation exceeded")
    results = [run_experiment(c) for c in configs]
    root = Path(os.getenv("RESULTS_DIR", "results"))
    summary = {"run_ids": [r["run_id"] for r in results], "analysis": analyze_sweep(results)}
    (root / "sweep-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    sweep_figure(results, root)
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="vfqec")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("experiment", choices=["E1", "E2", "E3", "E4", "E5"])
    run.add_argument("--backend", choices=["local", "aer", "bluequbit"], default=None)
    run.add_argument("--device", choices=["cpu", "gpu", "mps.cpu", "mps.gpu"], default=None)
    run.add_argument("--config", type=Path)
    for flag in (
        "shots",
        "evaluation-shots",
        "rounds",
        "iterations",
        "seed",
        "cadence",
        "max-shots",
        "max-remote-jobs",
    ):
        run.add_argument("--" + flag, type=int)
    run.add_argument("--max-credits", type=float)
    run.add_argument("--optimizer", choices=["spsa", "bayes"])
    run.add_argument("--basis", choices=["X", "Z"])
    run.add_argument(
        "--quick", action="store_true", help="Small smoke run; does not reproduce E1 defaults"
    )
    sweep = commands.add_parser("sweep")
    sweep.add_argument("plan", type=Path)
    report = commands.add_parser("report")
    report.add_argument("run_id")
    plan = commands.add_parser("plan")
    plan.add_argument("question")
    plan.add_argument("--max-shots", type=int, default=20_000_000)
    plan.add_argument("--output", type=Path, default=Path("plan.json"))
    commands.add_parser("ledger")
    hardware = commands.add_parser("hardware-template")
    hardware.add_argument("--output", type=Path, default=Path("results/hardware-template.qasm"))
    args = parser.parse_args()
    if args.command == "run":
        values = json.loads(args.config.read_text()) if args.config else {}
        for name in (
            "backend",
            "device",
            "shots",
            "evaluation_shots",
            "rounds",
            "iterations",
            "seed",
            "cadence",
            "optimizer",
            "basis",
            "max_shots",
            "max_remote_jobs",
            "max_credits",
        ):
            value = getattr(args, name)
            if value is not None:
                values[name] = value
        values.setdefault("backend", os.getenv("BACKEND", "local"))
        values.setdefault("device", "cpu")
        if args.quick:
            values.update(rounds=6, shots=128, evaluation_shots=128, iterations=4, cadence=3)
        if args.experiment == "E3":
            configs = [
                preset("E3", **(values | {"distance": d, "p": p, "eps_x": [0.12]}))
                for d in (3, 5, 7)
                for p in (0.001, 0.005, 0.02)
            ]
        elif args.experiment == "E4":
            configs = [
                preset("E4", **(values | {"cadence": k, "shots": s}))
                for k in (5, 15, 30)
                for s in (500, 2000)
            ]
        else:
            configs = [preset(args.experiment, **values)]
        if len(configs) == 1:
            results = [run_experiment(configs[0])]
        else:
            results = run_sweep(
                {
                    "runs": [c.model_dump() for c in configs],
                    "max_shots": values.get("max_shots", 20_000_000),
                    "max_credits": values.get("max_credits", 10),
                }
            )
        for result in results:
            print(
                json.dumps(
                    {
                        "run_id": result["run_id"],
                        "backend": result["backend"],
                        "shots_used": result["shots_used"],
                    }
                )
            )
    elif args.command == "sweep":
        for result in run_sweep(json.loads(args.plan.read_text())):
            print(result["run_id"])
    elif args.command == "report":
        row = Ledger().get(args.run_id)
        if row["status"] != "complete":
            raise ValueError("Only completed runs have reports")
        out = Path(os.getenv("RESULTS_DIR", "results")) / args.run_id
        print(generate_report(row["result"], out).resolve())
    elif args.command == "ledger":
        print(json.dumps(Ledger().list_runs(), indent=2))
    elif args.command == "plan":
        generated = PlannerAgent().plan(args.question, args.max_shots)
        args.output.write_text(json.dumps(generated, indent=2))
        print(args.output.resolve())
    elif args.command == "hardware-template":
        from qiskit import QuantumCircuit, qasm3

        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure(range(3), range(3))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(qasm3.dumps(circuit))
        args.output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "experiment": "E5",
                    "enabled": False,
                    "submitted": False,
                    "purpose": "GHZ readout/connectivity smoke template",
                    "backend": "unexecuted hardware template",
                    "shots": 100,
                    "note": "Manual hardware submission required; this is a GHZ smoke test.",
                },
                indent=2,
            )
        )
        print(args.output.resolve())


if __name__ == "__main__":
    main()
