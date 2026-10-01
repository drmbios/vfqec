"""Scientific invariants, not optimizer-specific golden numbers."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from vfqec.backends.local import LocalBackend
from vfqec.codes.stabilizer import make_code, surface
from vfqec.config import preset
from vfqec.decoders.matching import decoder_for
from vfqec.noise.model import HiddenPlant
from vfqec.physics.circuits import syndrome_circuit
from vfqec.physics.density import DensityEngine
from vfqec.physics.operators import initial_state, pauli
from vfqec.physics.trajectories import TrajectoryEngine


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_zero_noise_zero_logical_error(n, basis):
    c = preset("E1", distance=n, basis=basis, eps_x=[0], eps_z=[0], p=0, q=0)
    code, plant, backend = make_code(c), HiddenPlant(c), LocalBackend()
    rng = np.random.default_rng(7)
    if n <= 5:
        engine = DensityEngine(code, plant, backend)
        for r in range(3):
            error, fired, _ = engine.step(np.zeros(n), r, rng, 32)
            assert error == pytest.approx(0, abs=1e-12)
            assert fired == 0
            assert np.trace(engine.rho) == pytest.approx(1)
    else:
        engine = TrajectoryEngine(code, plant, backend, 16)
        for r in range(3):
            error, fired, _ = engine.step(np.zeros(n), r, rng)
            assert error == 0
            assert fired == 0


def test_oracle_matches_no_coherent_noise():
    noisy = preset("E1", q=0.03, p=0.005)
    clean = preset("E1", eps_x=[0], q=0.03, p=0.005)
    a = DensityEngine(make_code(noisy), HiddenPlant(noisy), LocalBackend())
    b = DensityEngine(make_code(clean), HiddenPlant(clean), LocalBackend())
    for t in range(12):
        a.step(np.array(noisy.eps_x), t, np.random.default_rng(t), 100)
        b.step(np.zeros(3), t, np.random.default_rng(t), 100)
        assert np.allclose(a.rho, b.rho, atol=1e-12)


def test_syndrome_cost_minimum_at_true_field():
    c = preset("E1", p=0, q=0)
    theta = np.array(c.eps_x)
    costs = []
    for shift in [-0.3, -0.15, 0, 0.15, 0.3]:
        engine = DensityEngine(make_code(c), HiddenPlant(c), LocalBackend())
        _, fired, checks = engine.step(theta + shift, 0, np.random.default_rng(4), 100000)
        costs.append(fired / checks)
    assert costs[2] == 0
    assert all(v > costs[2] for i, v in enumerate(costs) if i != 2)


def test_surface_stabilizers_and_single_error_recovery():
    code = surface()
    state = initial_state(code)
    assert np.allclose(code.check_matrix("X") @ code.check_matrix("Z").T % 2, 0)
    decoder = decoder_for(code)
    for axis in "XYZ":
        for q in range(code.n):
            damaged = pauli(state, code.n, axis, (q,))
            syndrome = np.array(
                [
                    [
                        int(np.vdot(damaged, pauli(damaged, code.n, a, s)).real < 0)
                        for a, s in code.checks
                    ]
                ],
                dtype=np.uint8,
            )
            x, z = decoder.decode(syndrome)
            for i in range(code.n):
                if x[0, i]:
                    damaged = pauli(damaged, code.n, "X", (i,))
                if z[0, i]:
                    damaged = pauli(damaged, code.n, "Z", (i,))
            assert abs(np.vdot(state, damaged)) == pytest.approx(1)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_zero_noise_and_oracle(basis):
    c = preset("E2", basis=basis, p=0, q=0, zz=0)
    engine = TrajectoryEngine(make_code(c), HiddenPlant(c), LocalBackend(), 32)
    for r in range(4):
        error, fired, _ = engine.step(HiddenPlant(c).oracle_theta(r), r, np.random.default_rng(r))
        assert error == 0
        assert fired == 0


def test_surface_circuit_has_17_qubits_and_zero_syndrome():
    code = surface()
    circuit = syndrome_circuit(code, measure=False)
    assert circuit.num_qubits == 17
    state = Statevector.from_instruction(circuit)
    probs = state.probabilities(list(range(9, 17)))
    assert probs[0] == pytest.approx(1, abs=1e-10)


def test_density_matches_trajectories_with_measurement_noise():
    c = preset("E1", p=0.04, q=0.03, eps_x=[0.2])
    exact = DensityEngine(make_code(c), HiddenPlant(c), LocalBackend())
    sampled = TrajectoryEngine(make_code(c), HiddenPlant(c), LocalBackend(), 20000)
    rng = np.random.default_rng(33)
    for r in range(3):
        p, _, _ = exact.step(np.zeros(3), r, rng, 1)
        observed, _, _ = sampled.step(np.zeros(3), r, rng)
        assert abs(p - observed) < 0.012
