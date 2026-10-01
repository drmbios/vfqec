import numpy as np
import pytest

from vfqec.optimizers.field import FieldOptimizer, Observation


@pytest.mark.parametrize("method", ["spsa", "bayes"])
def test_syndrome_interface_is_finite_shot_and_seeded(method):
    def oracle(theta, shots, seed):
        rng = np.random.default_rng(seed)
        cost = float(np.mean(np.sin((theta - 0.1) / 2) ** 2))
        return Observation(rng.binomial(shots, cost) / shots, shots, shots)

    a = FieldOptimizer(3, 4000, 20, 123, method).fit(oracle)
    b = FieldOptimizer(3, 4000, 20, 123, method).fit(oracle)
    assert np.array_equal(a.theta, b.theta)
    assert a.history == b.history
    assert a.shots_used == 4000 * (1 + (3 if method == "spsa" else 1) * 20)
    if method == "spsa":
        assert np.linalg.norm(a.theta - 0.1) < 0.1


def test_spsa_high_dimension_gain_remains_stable():
    target = np.linspace(-0.15, 0.15, 18)

    def oracle(theta, shots, seed):
        cost = float(np.mean(np.sin((theta - target) / 2) ** 2))
        rng = np.random.default_rng(seed)
        return Observation(rng.binomial(shots, cost) / shots, shots, shots)

    fit = FieldOptimizer(18, 4000, 40, 123).fit(oracle)
    assert np.linalg.norm(fit.theta - target) < 0.8 * np.linalg.norm(target)
