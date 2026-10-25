"""Reproducible held-out study; optimizer selection never sees logical diagnostics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from vfqec.agents.orchestrator import ExecutionAgent
from vfqec.backends.local import LocalBackend
from vfqec.codes.stabilizer import make_code
from vfqec.config import Config
from vfqec.ledger.store import Ledger
from vfqec.noise.model import HiddenPlant
from vfqec.optimizers.field import FieldOptimizer
from vfqec.optimizers.fourier import FourierOptimizer


def make_protocol() -> dict:
    """Fixed settings and seed allocations, generated before examining study outcomes."""
    cases = []

    def case(group, seed, **changes):
        distance = changes.get("distance", 3)
        rng = np.random.default_rng(seed + 700_000)
        fields = rng.uniform(-0.35, 0.35, distance).tolist()
        settings = dict(
            experiment="F1",
            window=1,
            rounds=60,
            shots=1000,
            evaluation_shots=4000,
            iterations=20,
            cadence=60,
            seed=seed,
            eps_x=fields,
            max_shots=10_000_000,
        )
        settings.update(changes)
        if settings.get("basis") == "X" and settings.get("code", "repetition") == "repetition":
            settings.update(eps_x=[0], eps_z=fields)
        config = Config.model_validate(settings)
        cases.append(
            {"id": f"{group}-{seed}", "group": group, "seed": seed, "config": config.model_dump()}
        )

    for seed in range(101, 105):
        case("development", seed)
    for seed in range(2001, 2025):
        case("heldout_static", seed)
    for seed in range(3001, 3005):
        case("phase_memory", seed, basis="X")
    for seed in range(3101, 3105):
        case("measurement_noise", seed, p=0.01, q=0.02)
    for distance in (5, 7):
        for seed in range(3201 + 10 * distance, 3203 + 10 * distance):
            case(
                f"distance_{distance}",
                seed,
                distance=distance,
                evaluation_shots=1000,
                rounds=30,
                cadence=30,
            )
    for seed in range(3401, 3405):
        case("drift", seed, drift_amplitude=0.15, drift_slope=0.001, drift_period=60, cadence=20)
    for basis in ("Z", "X"):
        for zz in (0.0, 0.08):
            seed = 4001 + (basis == "X") * 10 + int(zz > 0)
            case(
                f"surface_{basis}_zz{zz:.2f}",
                seed,
                code="surface",
                basis=basis,
                eps_x=[0.12],
                eps_z=[-0.08],
                zz=zz,
                p=0.002,
                shots=128,
                evaluation_shots=512,
                rounds=12,
                cadence=12,
                iterations=30,
            )
    return {
        "name": "F1 validation-gated Fourier calibration",
        "date": "2026-10-25",
        "version": 1,
        "primary_group": "heldout_static",
        "primary_endpoint": "final observed logical error",
        "noninferiority_margin_absolute": 0.001,
        "bootstrap_seed": 20261025,
        "bootstrap_replicates": 10000,
        "optimizer": {"fourier_passes": 4, "probe_radius": 0.7, "confidence": None},
        "budget": {
            "maximum_new_bluequbit_usd": 10,
            "zero_price_cpu_only": True,
            "maximum_cloud_jobs": 800,
        },
        "cloud_case_ids": ["heldout_static-2001", "phase_memory-3001"],
        "cases": cases,
        "notes": [
            "All optimizer probes use window=1, unlike the historical E1 window=2.",
            "Fourier validation is empirical acceptance, not a confidence certificate.",
            "Primary group includes every predeclared held-out seed, with no outcome filtering.",
            "Surface and distance panels are exploratory, reduced-shot extensions.",
            "No global novelty, hardware, threshold, or quantum advantage claim.",
        ],
    }


def protocol_digest(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def run_case(case: dict, protocol: dict, cloud_client=None) -> dict:
    """Fit blinded controllers, then reveal logical outcomes for every fixed comparator."""
    cloud = cloud_client is not None
    values = case["config"] | {"backend": "bluequbit" if cloud else "local"}
    config = Config.model_validate(values)
    ledger = Ledger()
    run_id = ledger.create(config.model_dump())
    ledger.finish(run_id, "running")
    out = Path(os.getenv("RESULTS_DIR", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        if cloud:
            from vfqec.backends.bluequbit import BlueQubitBackend

            backend = BlueQubitBackend(config, ledger, run_id, client=cloud_client)
        else:
            backend = LocalBackend()
        code, plant = make_code(config), HiddenPlant(config)
        executor = ExecutionAgent(config, code, plant, backend, ledger, run_id)
        surface = config.code == "surface"
        matched_budget = (1 + 3 * config.iterations) * config.shots
        names = ["spsa_budget", "fourier"]
        # The expensive original SPSA calibration is a reference only on 3/5-qubit density cases.
        if not surface and config.distance <= 5:
            names.insert(1, "spsa_reference")
        fits, updates = {}, {}
        for index, name in enumerate(names):
            schedule = []
            starts = range(0, config.rounds, config.cadence) if case["group"] == "drift" else [0]
            for calibration, start in enumerate(starts):
                seed = config.seed + 100 + 1000 * index + 10_000 * calibration
                if name == "fourier":
                    optimizer = FourierOptimizer(
                        config.dimensions,
                        config.shots,
                        matched_budget,
                        seed,
                        passes=1 if surface else protocol["optimizer"]["fourier_passes"],
                        probe_radius=protocol["optimizer"]["probe_radius"],
                    )
                else:
                    optimizer = FieldOptimizer(
                        config.dimensions,
                        4000 if name == "spsa_reference" else config.shots,
                        50 if name == "spsa_reference" else config.iterations,
                        seed,
                    )
                initial = np.array(schedule[-1]["theta"]) if schedule else None
                fit = optimizer.fit(executor.syndrome_oracle(start), initial)
                schedule.append(
                    {
                        "round": start,
                        "theta": fit.theta.tolist(),
                        "shots_used": fit.shots_used,
                        "history": fit.history,
                    }
                )
            fits[name] = schedule[0]
            updates[name] = schedule
        schedules = {
            "unencoded": lambda _: np.zeros(config.dimensions),
            "standard": lambda _: np.zeros(config.dimensions),
            "twirled": lambda _: np.zeros(config.dimensions),
            "oracle": plant.oracle_theta,
        }
        for name in names:
            schedules[name] = lambda _, name=name: np.array(fits[name]["theta"])
        if case["group"] == "drift":
            for name, label in (("spsa_budget", "adaptive_spsa"), ("fourier", "adaptive_fourier")):
                schedules[label] = lambda t, name=name: np.array(
                    updates[name][t // config.cadence]["theta"]
                )
        methods = {}
        for index, (name, schedule) in enumerate(schedules.items()):
            seed = config.seed + 50_000 + index
            executor.reserve(config.evaluation_shots * config.rounds)
            trace = executor.simulate(
                schedule,
                config.rounds,
                config.evaluation_shots,
                seed,
                twirl=name == "twirled",
                unencoded=name == "unencoded",
            )
            methods[name] = {
                "errors": trace.errors.tolist(),
                "shots": config.evaluation_shots,
                "logical_error": (trace.errors / config.evaluation_shots).tolist(),
                "exact_probability": trace.exact_probability.tolist()
                if trace.exact_probability is not None
                else None,
                "engine": trace.engine,
                "seed": seed,
            }
        # These diagnostics are populated only after every optimizer has returned.
        result = {
            "case": case,
            "run_id": run_id,
            "commit": ledger.get(run_id)["commit"],
            "protocol_sha256": protocol_digest(protocol),
            "backend": backend.label,
            "fits": fits,
            "updates": updates,
            "methods": methods,
            "true_fields": [plant.oracle_theta(t).tolist() for t in range(config.rounds)],
            "shots_used": executor.used,
            "remote_jobs": ledger.run_jobs(run_id),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        ledger.finish(run_id, "complete", result)
        return result
    except Exception as exc:
        ledger.finish(
            run_id,
            "failed",
            {
                "error_type": type(exc).__name__,
                "message": "Research case stopped; inspect private log",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["protocol", "development", "heldout", "extensions", "cloud", "report"]
    )
    parser.add_argument("--root", type=Path, default=Path("results/F1-campaign"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if args.stage == "protocol":
        if protocol_path.exists():
            raise ValueError("Protocol exists; preserve frozen settings")
        protocol_path.write_text(json.dumps(make_protocol(), indent=2, allow_nan=False))
        print(protocol_digest(make_protocol()), flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    if args.stage == "report":
        from vfqec.research_analysis import write_campaign_report

        results = [json.loads(p.read_text()) for p in sorted((root / "local").glob("*.json"))]
        print(json.dumps(write_campaign_report(results, root), indent=2), flush=True)
        return
    cloud_client = None
    if args.stage == "cloud":
        import logging

        import bluequbit

        from vfqec.backends.campaign_budget import CampaignBudgetClient

        load_dotenv()
        logging.getLogger("bluequbit-python-sdk").setLevel(logging.CRITICAL)
        credential = os.getenv("BLUEQUBIT_API_TOKEN")
        if not credential:
            raise ValueError("Configure BlueQubit credential locally")
        raw = bluequbit.init(api_token=credential, execution_mode="cloud")
        cloud_client = CampaignBudgetClient(
            raw, path=root / "cloud-budget.sqlite3", cap_usd="10", max_jobs=800
        )
    directory = root / ("cloud" if cloud_client else "local")
    directory.mkdir(exist_ok=True)
    for case in protocol["cases"]:
        group = case["group"]
        selected = {
            "development": group == "development",
            "heldout": group == "heldout_static",
            "extensions": group not in ("development", "heldout_static"),
            "cloud": case["id"] in protocol["cloud_case_ids"],
        }[args.stage]
        path = directory / (case["id"] + ".json")
        if not selected or path.exists():
            continue
        print(json.dumps({"event": "started", "case": case["id"]}), flush=True)
        result = run_case(case, protocol, cloud_client)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(
            json.dumps(
                {
                    "event": "complete",
                    "case": case["id"],
                    "run_id": result["run_id"],
                    "final": {k: v["errors"][-1] for k, v in result["methods"].items()},
                }
            ),
            flush=True,
        )
        if cloud_client:
            (root / "cloud-budget.json").write_text(json.dumps(cloud_client.summary(), indent=2))


if __name__ == "__main__":
    main()
