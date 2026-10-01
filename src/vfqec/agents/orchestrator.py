"""Agent classes and the execution firewall around the hidden plant."""

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit.quantum_info import Statevector

from vfqec.agents.llm import LLM
from vfqec.config import Config, preset
from vfqec.optimizers.field import Observation
from vfqec.physics.circuits import syndrome_circuit
from vfqec.physics.density import DensityEngine
from vfqec.physics.trajectories import TrajectoryEngine


class PlannerAgent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    def plan(self, question: str, max_shots: int, past: list[dict] | None = None) -> dict:
        """Prune previously measured saturated regions; enforce aggregate shot cap."""
        advice = self.llm.advise(
            "planner",
            {"question": question, "max_shots": max_shots},
            {
                "code": "repetition",
                "distance": 3,
                "p": [0.001, 0.003, 0.01],
                "q": 0.0,
                "eps": 0.12,
                "drift": 0.0,
                "shots": 1000,
                "optimizer": "spsa",
            },
        )
        configs, pruned, budget = [], [], 0
        try:
            values = advice.get("p", [0.001, 0.003, 0.01])
            if not isinstance(values, list) or len(values) > 20:
                raise ValueError("Invalid sweep size")
            candidates = [
                preset(
                    "E3",
                    code=advice.get("code", "repetition"),
                    distance=advice.get("distance", 3),
                    p=p,
                    q=advice.get("q", 0),
                    eps_x=[advice.get("eps", 0.12)],
                    drift_amplitude=advice.get("drift", 0),
                    shots=advice.get("shots", 1000),
                    optimizer=advice.get("optimizer", "spsa"),
                    max_shots=max_shots,
                )
                for p in values
            ]
        except (ValueError, TypeError):
            candidates = [preset("E3", p=p, max_shots=max_shots) for p in (0.001, 0.003, 0.01)]
        for config in candidates:
            saturated = any(
                r.get("p") == config.p
                and r.get("distance") == config.distance
                and r.get("basis", "Z") == config.basis
                and r.get("logical_error", 0) > 0.45
                for r in (past or [])
            )
            if saturated or budget + config.shot_bound > max_shots:
                pruned.append(
                    {"p": config.p, "reason": "saturated" if saturated else "shot budget"}
                )
                continue
            configs.append(config.model_dump())
            budget += config.shot_bound
        return {
            "question": question,
            "max_shots": max_shots,
            "runs": configs,
            "shot_bound": budget,
            "pruned": pruned,
            "advice": advice,
        }


class CircuitBuilderAgent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    def build_and_validate(self, code) -> tuple:
        from vfqec.codes.stabilizer import repetition

        tiny = repetition(3, code.basis)
        circuit = syndrome_circuit(tiny, measure=False)
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities(list(range(tiny.n, tiny.n + tiny.m)))
        if not np.isclose(probabilities[0], 1):
            raise ValueError("Tiny noiseless syndrome circuit failed validation")
        x, z = code.check_matrix("X"), code.check_matrix("Z")
        if np.any((x @ z.T) % 2):
            raise ValueError("Noncommuting stabilizers")
        advice = self.llm.advise(
            "circuit-builder",
            {
                "code": code.name,
                "data_qubits": code.n,
                "ancillas": code.m,
                "tiny_validation": "passed",
            },
            {"action": "execute"},
        )
        return syndrome_circuit(code), advice


@dataclass
class Trace:
    errors: np.ndarray
    syndrome_fired: np.ndarray
    syndrome_checks: np.ndarray
    exact_probability: np.ndarray | None
    engine: str


class ExecutionAgent:
    """Optimizer-facing facade returns syndrome counts, never logical observables."""

    def __init__(self, config: Config, code, plant, backend, ledger, run_id: str, llm=None):
        self.config, self.code, self.plant, self.backend = config, code, plant, backend
        self.ledger, self.run_id = ledger, run_id
        self.llm = llm or LLM()
        self.cache = {}
        self.used = 0
        self.ledger.event(
            run_id,
            {
                "type": "execution_advice",
                "advice": self.llm.advise(
                    "execution",
                    {"backend": backend.label, "shot_bound": config.shot_bound},
                    {"action": "execute", "retry_policy": "resume known remote job IDs only"},
                ),
            },
        )

    def reserve(self, shots: int) -> None:
        if self.used + shots > self.config.max_shots:
            raise RuntimeError("Experiment shot budget exhausted")
        self.used += shots

    def _density(self, unencoded: bool) -> bool:
        return unencoded or (self.code.name == "repetition" and self.code.n <= 5)

    def simulate(
        self,
        schedule: Callable[[int], np.ndarray],
        rounds: int,
        shots: int,
        seed: int,
        start: int = 0,
        twirl: bool = False,
        unencoded: bool = False,
        progress: bool = False,
    ) -> Trace:
        rng = np.random.default_rng(seed)
        fired, checks, errors = (np.zeros(rounds, dtype=np.int64) for _ in range(3))
        density = self._density(unencoded)
        exact = np.zeros(rounds) if density else None
        if density:
            engine = DensityEngine(self.code, self.plant, self.backend, unencoded)
            for r in range(rounds):
                prob, fired[r], checks[r] = engine.step(
                    schedule(start + r), start + r, rng, shots, twirl
                )
                exact[r] = prob
                errors[r] = rng.binomial(shots, prob)
                if progress:
                    self.ledger.event(self.run_id, {"type": "round", "round": r + 1})
        else:
            for offset in range(0, shots, self.config.batch_size):
                count = min(self.config.batch_size, shots - offset)
                engine = TrajectoryEngine(self.code, self.plant, self.backend, count)
                for r in range(rounds):
                    prob, f, c = engine.step(schedule(start + r), start + r, rng, twirl)
                    errors[r] += round(prob * count)
                    fired[r] += f
                    checks[r] += c
                if progress:
                    self.ledger.event(
                        self.run_id,
                        {"type": "batch", "completed_shots": offset + count, "total_shots": shots},
                    )
        return Trace(
            errors,
            fired,
            checks,
            exact,
            "exact density matrix" if density else "sampled state-vector trajectories",
        )

    def syndrome_oracle(self, start: int = 0) -> Callable:
        """Capability passed to FieldOptimizer; closures are a design boundary, not a sandbox."""

        def observe(theta: np.ndarray, shots: int, seed: int) -> Observation:
            key = hashlib.sha256(
                json.dumps([theta.tolist(), shots, seed, start]).encode()
            ).hexdigest()
            if key in self.cache:
                return self.cache[key]
            self.reserve(shots)
            rng = np.random.default_rng(seed)
            fired, checks = 0, 0
            # Allocate a total of 'shots' across independently prepared prefix circuits.
            # The endpoint syndrome of each prefix measures the short-window mean cost.
            window = min(self.config.window, shots)
            for prefix in range(1, window + 1):
                count = shots // window + int(prefix <= shots % window)
                trace = self.simulate(
                    lambda _: theta, prefix, count, int(rng.integers(2**31)), start=start
                )
                fired += int(trace.syndrome_fired[-1])
                checks += int(trace.syndrome_checks[-1])
            observation = Observation(fired / checks, shots, checks)
            self.cache[key] = observation
            return observation

        return observe


class AnalystAgent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    def analyze(self, result: dict) -> dict:
        from vfqec.report.analysis import analyze

        analysis = analyze(result)
        analysis["advice"] = self.llm.advise(
            "analyst",
            {"fits": analysis["fits"], "anomalies": analysis["anomalies"]},
            {"action": "review", "reruns": analysis["reruns"]},
        )
        return analysis
