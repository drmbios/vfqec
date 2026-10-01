"""Six-method experiment runner. Diagnostics are revealed only after optimization."""

import json
import os
from pathlib import Path

import numpy as np
from qiskit import qasm3

from vfqec.agents.orchestrator import AnalystAgent, CircuitBuilderAgent, ExecutionAgent
from vfqec.backends.aer import AerBackend
from vfqec.backends.local import LocalBackend
from vfqec.codes.stabilizer import make_code
from vfqec.config import Config
from vfqec.ledger.store import Ledger
from vfqec.noise.model import HiddenPlant
from vfqec.optimizers.field import FieldOptimizer

METHODS = ("unencoded", "standard", "twirled", "calibrated", "adaptive", "oracle")


def run_experiment(config: Config, ledger: Ledger | None = None, run_id: str | None = None) -> dict:
    """Execute one complete experiment and persist failures as well as successes."""
    ledger = ledger or Ledger()
    run_id = run_id or ledger.create(config.model_dump())
    out = Path(os.getenv("RESULTS_DIR", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    ledger.finish(run_id, "running")
    try:
        if config.shot_bound > config.max_shots:
            raise ValueError(
                f"Plan requires at most {config.shot_bound} shots; budget is {config.max_shots}"
            )
        if config.backend == "aer" and config.aer_method == "stabilizer":
            raise ValueError(
                "E1-E4 require arbitrary rotations. Stabilizer mode is a Clifford-only API"
            )
        if config.backend == "bluequbit":
            from vfqec.backends.bluequbit import BlueQubitBackend

            backend = BlueQubitBackend(config, ledger, run_id)
        elif config.backend == "aer":
            backend = AerBackend(config.aer_method, config.seed)
        else:
            backend = LocalBackend()
        code = make_code(config)
        plant = HiddenPlant(config)
        circuit, advice = CircuitBuilderAgent().build_and_validate(code)
        (out / "syndrome-circuit.qasm").write_text(qasm3.dumps(circuit))
        ledger.event(
            run_id, {"type": "circuit_validated", "advice": advice, "qubits": circuit.num_qubits}
        )
        executor = ExecutionAgent(config, code, plant, backend, ledger, run_id)
        histories, updates = [], []
        theta = None
        for calibration, start in enumerate(range(0, config.rounds, config.cadence)):

            def report(row, start=start):
                ledger.event(run_id, {"type": "optimizer", "calibration_round": start, **row})

            optimizer = FieldOptimizer(
                config.dimensions,
                config.shots,
                config.iterations,
                config.seed + 100 + calibration,
                config.optimizer,
                report,
            )
            fit = optimizer.fit(executor.syndrome_oracle(start), theta)
            theta = fit.theta
            updates.append({"round": start, "theta": theta.tolist(), "shots": fit.shots_used})
            histories.extend({"calibration_round": start, **row} for row in fit.history)
        initial_theta = np.array(updates[0]["theta"])

        def adaptive(t):
            return np.array(updates[t // config.cadence]["theta"])

        result = {
            "run_id": run_id,
            "config": config.model_dump(),
            "backend": backend.label,
            "commit": ledger.get(run_id)["commit"],
            "methods": {},
            "optimizer": histories,
            "updates": updates,
            "notes": [
                "Phenomenological data noise; ideal gates, preparation and terminal readout.",
                "Spatial decoding each round; q>0 is not fault-tolerant spacetime decoding.",
                "Calibration uses fresh encoded probes; drift time freezes during calibration.",
                "Oracle cancels single-qubit fields only; ZZ and stochastic noise remain.",
                "Endpoint curves share simulated trajectories and have correlated time points.",
                "Zero observed failures have nonzero confidence upper bounds.",
            ],
        }
        for index, method in enumerate(METHODS):
            ledger.event(run_id, {"type": "method_started", "method": method})
            schedule = {
                "calibrated": lambda _: initial_theta,
                "adaptive": adaptive,
                "oracle": plant.oracle_theta,
            }.get(method, lambda _: np.zeros(config.dimensions))
            executor.reserve(config.evaluation_shots * config.rounds)
            trace = executor.simulate(
                schedule,
                config.rounds,
                config.evaluation_shots,
                config.seed + 10_000 + index,
                twirl=method == "twirled",
                unencoded=method == "unencoded",
                progress=True,
            )
            result["methods"][method] = {
                "errors": trace.errors.tolist(),
                "shots": config.evaluation_shots,
                "logical_error": (trace.errors / config.evaluation_shots).tolist(),
                "syndrome_fired": trace.syndrome_fired.tolist(),
                "syndrome_checks": trace.syndrome_checks.tolist(),
                "exact_probability": trace.exact_probability.tolist()
                if trace.exact_probability is not None
                else None,
                "engine": trace.engine,
                "seed": config.seed + 10_000 + index,
                "backend": backend.label,
            }
            ledger.event(
                run_id,
                {
                    "type": "method_completed",
                    "method": method,
                    "final_logical_error": result["methods"][method]["logical_error"][-1],
                },
            )
        # Evaluation-only ground truth. Never enters optimizer history, prompts, or selection.
        result["true_fields"] = [plant.oracle_theta(t).tolist() for t in range(config.rounds)]
        result["shots_used"] = executor.used
        result["remote_jobs"] = ledger.run_jobs(run_id)
        result["analysis"] = AnalystAgent().analyze(result)
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        from vfqec.report.generate import generate_report

        generate_report(result, out)
        ledger.finish(run_id, "complete", result)
        ledger.event(run_id, {"type": "complete", "output_dir": str(out.resolve())})
        return result
    except Exception as exc:
        # Keep secrets and provider HTTP bodies out of the persistent ledger.
        failure = {
            "error_type": type(exc).__name__,
            "message": "Run failed; inspect local worker log",
        }
        ledger.finish(run_id, "failed", failure)
        ledger.event(run_id, {"type": "failed", **failure})
        raise
