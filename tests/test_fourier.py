import numpy as np
import pytest

from vfqec.optimizers.field import Observation
from vfqec.optimizers.fourier import FourierOptimizer


def test_fourier_recovers_bounded_first_harmonic_and_accounts_every_probe():
    target = np.array([0.12, -0.2, 0.7])
    calls = []

    def oracle(theta, shots, seed):
        calls.append((theta.copy(), shots, seed))
        cost = float(np.mean(np.sin((theta - target) / 2) ** 2))
        return Observation(cost, shots, 2 * shots)

    fit = FourierOptimizer(3, 1000, 15_999, 14, passes=4).fit(oracle)
    assert np.allclose(fit.theta, target, atol=1e-12)
    assert fit.shots_used == 15_000
    assert len(calls) == len(fit.history) == 15
    assert len({row[2] for row in calls}) == 15
    assert all(np.max(np.abs(row[0])) <= 0.7 for row in calls)
    assert [row["shots_used"] for row in fit.history] == list(range(1000, 15_001, 1000))


def test_fourier_finite_shot_reproducibility_and_arbitrary_dimension():
    target = np.linspace(-0.2, 0.2, 6)

    def oracle(theta, shots, seed):
        p = float(np.mean(np.sin((theta - target) / 2) ** 2))
        cost = np.random.default_rng(seed).binomial(shots, p) / shots
        return Observation(cost, shots, shots)

    a = FourierOptimizer(6, 20_000, 600_000, 42).fit(oracle)
    b = FourierOptimizer(6, 20_000, 600_000, 42).fit(oracle)
    assert np.array_equal(a.theta, b.theta)
    assert a.history == b.history
    assert np.linalg.norm(a.theta - target) < np.linalg.norm(target) * 0.5
    assert a.shots_used == 600_000


def test_validation_rejects_misleading_fit_and_callback_records_decision():
    recorded = []
    counter = 0

    def oracle(theta, shots, seed):
        nonlocal counter
        counter += 1
        if counter <= 3:
            # Deliberately deceptive fit probes; the fresh candidate check is worse.
            cost = float(0.1 + 0.1 * np.sin((theta[0] - 0.3) / 2) ** 2)
        else:
            cost = 0.1 if counter == 4 else 0.2
        return Observation(cost, shots, shots)

    fit = FourierOptimizer(1, 100, 500, 1, callback=recorded.append).fit(oracle)
    assert fit.theta.tolist() == [0.0]
    assert fit.history[-1]["accepted"] is False
    assert recorded == fit.history
    assert fit.shots_used == 500


def test_confidence_gate_uses_probe_shots_not_correlated_check_count():
    counter = 0

    def oracle(theta, shots, seed):
        nonlocal counter
        counter += 1
        if counter <= 3:
            cost = float(0.1 + 0.1 * np.sin((theta[0] - 0.3) / 2) ** 2)
        else:
            cost = 0.1 if counter == 4 else 0.09
        return Observation(cost, shots, 1000 * shots)

    fit = FourierOptimizer(1, 1000, 5000, 5, confidence=0.05).fit(oracle)
    assert fit.history[-1]["validation_radius"] > 0.01
    assert fit.history[-1]["accepted"] is False
    assert fit.theta.tolist() == [0.0]


def test_insufficient_budget_does_not_call_oracle_or_mutate_initial():
    initial = np.array([0.1, -0.2])

    def forbidden(*args):
        raise AssertionError("An unfunded coordinate must not be started")

    fit = FourierOptimizer(2, 100, 499, 1).fit(forbidden, initial)
    assert fit.shots_used == 0
    assert fit.history == []
    assert np.array_equal(fit.theta, initial)
    fit.theta[0] = 0.4
    assert initial.tolist() == [0.1, -0.2]


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_cost_is_rejected(bad):
    with pytest.raises(ValueError, match="Invalid syndrome"):
        FourierOptimizer(1, 100, 500, 1).fit(lambda *_: Observation(bad, 100, 100))


def test_invalid_shot_accounting_and_initial_controls_are_rejected():
    with pytest.raises(ValueError, match="shot accounting"):
        FourierOptimizer(1, 100, 500, 1).fit(lambda *_: Observation(0.1, 99, 100))
    with pytest.raises(ValueError, match="controls"):
        FourierOptimizer(1, 100, 500, 1).fit(lambda *_: None, np.array([0.8]))
