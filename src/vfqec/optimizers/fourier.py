"""Finite-shot, syndrome-only Fourier coordinate calibration.

The three-point sinusoidal coordinate fit is the known Rotosolve idea, not a
claim of a new optimization algorithm. Its first-harmonic premise applies when
each control angle occurs once in a single-round probe circuit. Multi-round
costs generally contain higher harmonics and need not satisfy that premise.
"""

from math import pi
from typing import Callable

import numpy as np

from vfqec.optimizers.field import Fit, SyndromeOracle


class FourierOptimizer:
    """Fit bounded coordinate slices, then validate proposed changes independently.

    Every attempted coordinate consumes exactly five observations: three fit
    anchors and fresh baseline/candidate validation shots. The default accepts
    a lower validation cost, without claiming statistical significance. Optional
    ``confidence`` is a familywise error probability for conservative Hoeffding
    acceptance. Its sample size is independent probe shots, never the number of
    potentially correlated stabilizer checks within a shot.
    """

    def __init__(
        self,
        dimensions: int,
        shots: int,
        max_shots: int,
        seed: int,
        passes: int = 2,
        probe_radius: float = 0.7,
        confidence: float | None = None,
        callback: Callable | None = None,
    ):
        for name, value, minimum in (
            ("dimensions", dimensions, 1),
            ("shots", shots, 1),
            ("max_shots", max_shots, 0),
            ("passes", passes, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(probe_radius) or not 0 < probe_radius <= 0.7:
            raise ValueError("probe_radius must be in (0, 0.7]")
        if confidence is not None and not 0 < confidence < 1:
            raise ValueError("confidence must be an error probability in (0, 1)")
        self.dimensions, self.shots, self.max_shots = dimensions, shots, max_shots
        self.seed, self.passes, self.probe_radius = seed, passes, probe_radius
        self.confidence, self.callback = confidence, callback

    def fit(self, oracle: SyndromeOracle, initial: np.ndarray | None = None) -> Fit:
        theta = np.zeros(self.dimensions) if initial is None else np.array(initial, dtype=float)
        if theta.shape != (self.dimensions,) or not np.isfinite(theta).all():
            raise ValueError("initial must be a finite vector with the requested dimensions")
        if np.any(np.abs(theta) > 0.7):
            raise ValueError("initial controls must lie in [-0.7, 0.7]")
        rng = np.random.default_rng(self.seed)
        history, used_seeds, used = [], set(), 0
        steps = min(self.passes * self.dimensions, self.max_shots // (5 * self.shots))
        if 5 * steps >= 2**31:
            raise ValueError("Too many observations for the distinct-seed schedule")
        # All observations use distinct seeds, including baseline/candidate validation.
        anchors = np.array([-self.probe_radius, 0.0, self.probe_radius])
        design = np.column_stack([np.ones(3), np.cos(anchors), np.sin(anchors)])
        radius = (
            float(np.sqrt(np.log(4 * steps / self.confidence) / (2 * self.shots)))
            if self.confidence is not None and steps
            else 0.0
        )

        def evaluate(point, phase, step, coordinate, defer_callback=False):
            nonlocal used
            seed = int(rng.integers(2**31))
            while seed in used_seeds:
                seed = int(rng.integers(2**31))
            used_seeds.add(seed)
            observation = oracle(point.copy(), self.shots, seed)
            if (
                not np.isfinite(observation.cost)
                or not 0 <= observation.cost <= 1
                or observation.shots != self.shots
                or observation.checks <= 0
            ):
                raise ValueError("Invalid syndrome observation or shot accounting")
            used += observation.shots
            row = {
                "step": step,
                "phase": phase,
                "coordinate": coordinate,
                "pass": (step - 1) // self.dimensions + 1,
                "cost": float(observation.cost),
                "theta": point.tolist(),
                "shots": observation.shots,
                "checks": observation.checks,
                "seed": seed,
                "shots_used": used,
            }
            history.append(row)
            if self.callback and not defer_callback:
                self.callback(row.copy())
            return row

        for step in range(1, steps + 1):
            coordinate = (step - 1) % self.dimensions
            costs = []
            for anchor in anchors:
                point = theta.copy()
                point[coordinate] = anchor
                costs.append(evaluate(point, "fourier_probe", step, coordinate)["cost"])
            coefficients = np.linalg.solve(design, costs)
            _, cosine, sine = coefficients
            # Minima of a + b cos(x) + c sin(x), plus both constrained boundaries.
            minimum = float(np.arctan2(sine, cosine) + pi)
            choices = [-0.7, 0.7, float(theta[coordinate])]
            choices.extend(minimum + 2 * pi * k for k in range(-2, 2))
            choices = [x for x in choices if -0.7 <= x <= 0.7]

            def fitted_cost(x):
                return cosine * np.cos(x) + sine * np.sin(x)

            candidate = theta.copy()
            candidate[coordinate] = min(choices, key=fitted_cost)
            baseline = evaluate(theta, "validation_baseline", step, coordinate)
            trial = evaluate(candidate, "candidate", step, coordinate, defer_callback=True)
            accepted = bool(trial["cost"] + radius < baseline["cost"] - radius)
            trial.update(
                accepted=accepted,
                baseline_cost=baseline["cost"],
                validation_radius=radius,
                confidence=self.confidence,
                fitted_coefficients=coefficients.tolist(),
            )
            if self.callback:
                self.callback(trial.copy())
            if accepted:
                theta = candidate
        return Fit(theta.copy(), history, used)
