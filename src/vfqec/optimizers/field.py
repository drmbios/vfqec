"""Black-box syndrome-only optimizers. This module has no access to the hidden plant."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import optuna


@dataclass(frozen=True)
class Observation:
    cost: float
    shots: int
    checks: int


class SyndromeOracle(Protocol):
    def __call__(self, theta: np.ndarray, shots: int, seed: int) -> Observation: ...


@dataclass
class Fit:
    theta: np.ndarray
    history: list[dict]
    shots_used: int


class FieldOptimizer:
    """Bounded SPSA or Bayesian TPE search using only finite-shot syndrome observations."""

    def __init__(
        self,
        dimensions: int,
        shots: int,
        iterations: int,
        seed: int,
        method: str = "spsa",
        callback: Callable | None = None,
    ):
        self.dimensions, self.shots, self.iterations = dimensions, shots, iterations
        self.seed, self.method, self.callback = seed, method, callback

    def fit(self, oracle: SyndromeOracle, initial: np.ndarray | None = None) -> Fit:
        rng = np.random.default_rng(self.seed)
        theta = np.zeros(self.dimensions) if initial is None else np.array(initial, copy=True)
        history, used = [], 0

        def evaluate(point, phase, step):
            nonlocal used
            seed = int(rng.integers(2**31))
            observation = oracle(point.copy(), self.shots, seed)
            if not 0 <= observation.cost <= 1 or observation.shots != self.shots:
                raise ValueError("Invalid syndrome observation or shot accounting")
            used += observation.shots
            row = {
                "step": step,
                "phase": phase,
                "cost": observation.cost,
                "theta": point.tolist(),
                "shots": observation.shots,
                "checks": observation.checks,
                "seed": seed,
                "shots_used": used,
            }
            history.append(row)
            if self.callback:
                self.callback(row)
            return observation.cost

        evaluate(theta, "initial", 0)
        if self.method == "spsa":
            average = []
            for k in range(self.iterations):
                delta = rng.choice([-1.0, 1.0], self.dimensions)
                perturbation = 0.15 / (k + 1) ** 0.101
                plus = evaluate(np.clip(theta + perturbation * delta, -0.7, 0.7), "plus", k + 1)
                minus = evaluate(np.clip(theta - perturbation * delta, -0.7, 0.7), "minus", k + 1)
                gradient = (plus - minus) / (2 * perturbation) * delta
                # Bound the gain as dimension grows: SPSA direction variance grows with dimension.
                rate = 1.2 * np.sqrt(3 / self.dimensions) / (1 + k / 15) ** 0.602
                theta = np.clip(theta - rate * gradient, -0.7, 0.7)
                evaluate(theta, "candidate", k + 1)
                if k >= self.iterations // 2:
                    average.append(theta.copy())
            theta = np.mean(average, axis=0)
        elif self.method == "bayes":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(
                seed=self.seed, n_startup_trials=min(10, max(2, self.iterations // 3))
            )
            study = optuna.create_study(sampler=sampler, direction="minimize")
            study.enqueue_trial({f"theta_{i}": float(v) for i, v in enumerate(theta)})

            def objective(trial):
                point = np.array(
                    [trial.suggest_float(f"theta_{i}", -0.7, 0.7) for i in range(self.dimensions)]
                )
                return evaluate(point, "candidate", trial.number + 1)

            study.optimize(objective, n_trials=self.iterations)
            theta = np.array([study.best_params[f"theta_{i}"] for i in range(self.dimensions)])
        else:
            raise ValueError("Optimizer must be spsa or bayes")
        return Fit(theta, history, used)
