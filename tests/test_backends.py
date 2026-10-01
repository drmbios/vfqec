from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_density_matrix

from vfqec.backends.aer import AerBackend
from vfqec.backends.bluequbit import BlueQubitBackend
from vfqec.backends.local import LocalBackend
from vfqec.config import preset
from vfqec.ledger.store import Ledger
from vfqec.noise.model import Gate


def test_numpy_aer_density_agree():
    rho = np.asarray(random_density_matrix(8, seed=73))
    gates = [Gate("X", (0,), 0.12), Gate("Z", (1,), -0.2), Gate("Z", (1, 2), 0.07)]
    expected = LocalBackend().evolve_density(rho, gates, 3)
    actual = AerBackend().evolve_density(rho, gates, 3)
    assert np.allclose(expected, actual, atol=1e-10)
    assert np.trace(actual) == pytest.approx(1)


def test_aer_stabilizer_clifford():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(range(2), range(2))
    counts = AerBackend("stabilizer", 3).run_counts(circuit, 200)
    assert set(counts) == {"00", "11"}


class FakeBlueQubit:
    """Deterministic SDK contract double; explicitly not a remote integration test."""

    def __init__(self):
        self.calls = 0
        self.waits = 0

    def estimate(self, circuit, device):
        return SimpleNamespace(estimated_cost=0.001)

    def run(self, circuit, **kwargs):
        self.calls += 1
        self.state = np.asarray(Statevector.from_instruction(circuit))
        return SimpleNamespace(job_id="contract-test-job")

    def wait(self, job_id, timeout):
        self.waits += 1
        return SimpleNamespace(ok=True, cost=0.001, get_statevector=lambda: self.state)


def test_bluequbit_choi_ordering_dedupe_and_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    ledger = Ledger()
    config = preset("E1", backend="bluequbit")
    run_id = ledger.create(config.model_dump())
    client = FakeBlueQubit()
    backend = BlueQubitBackend(config, ledger, run_id, client)
    rho = np.asarray(random_density_matrix(8, seed=3))
    gates = [Gate("X", (2,), -0.17), Gate("Z", (0,), 0.12), Gate("Z", (0, 1), 0.23)]
    a = backend.evolve_density(rho, gates, 3)
    b = LocalBackend().evolve_density(rho, gates, 3)
    assert np.allclose(a, b)
    backend.evolve_density(rho, gates, 3)
    assert client.calls == 1
    assert ledger.run_jobs(run_id)[0]["job_id"] == "contract-test-job"


def test_bluequbit_budget_checked_before_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    ledger = Ledger()
    config = preset("E1", backend="bluequbit", max_credits=0)
    client = FakeBlueQubit()
    backend = BlueQubitBackend(config, ledger, ledger.create(config.model_dump()), client)
    with pytest.raises(RuntimeError, match="credit budget"):
        backend.evolve_density(np.eye(2) / 2, [Gate("X", (0,), 0.1)], 1)
    assert client.calls == 0


def test_bluequbit_rate_limit_retries_without_duplicate_job(tmp_path, monkeypatch):
    from bluequbit.exceptions import BQAPIError

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr("vfqec.backends.bluequbit.time.sleep", lambda _: None)
    ledger = Ledger()
    config = preset("E1", backend="bluequbit")

    class Limited(FakeBlueQubit):
        attempts = 0

        def run(self, circuit, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise BQAPIError(429, "Rate limited", "test-request")
            return super().run(circuit, **kwargs)

    client = Limited()
    run_id = ledger.create(config.model_dump())
    backend = BlueQubitBackend(config, ledger, run_id, client)
    backend.evolve_density(np.eye(2) / 2, [Gate("X", (0,), 0.1)], 1)
    assert client.attempts == 2
    assert client.calls == 1
    assert ledger.run_jobs(run_id)[0]["status"] == "complete"


def test_bluequbit_ambiguous_submission_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    ledger = Ledger()
    config = preset("E1", backend="bluequbit")

    class Ambiguous(FakeBlueQubit):
        def run(self, circuit, **kwargs):
            self.calls += 1
            raise TimeoutError("Submission outcome unknown")

    client = Ambiguous()
    backend = BlueQubitBackend(config, ledger, ledger.create(config.model_dump()), client)
    with pytest.raises(TimeoutError):
        backend.evolve_density(np.eye(2) / 2, [Gate("X", (0,), 0.1)], 1)
    assert client.calls == 1


def test_bluequbit_nine_data_qubits_use_small_remote_probes(tmp_path, monkeypatch):
    from vfqec.codes.stabilizer import make_code
    from vfqec.noise.model import HiddenPlant
    from vfqec.physics.operators import initial_state

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    ledger = Ledger()
    config = preset("E2", backend="bluequbit")

    class Capped(FakeBlueQubit):
        def run(self, circuit, **kwargs):
            assert circuit.num_qubits <= 16
            return super().run(circuit, **kwargs)

    client = Capped()
    backend = BlueQubitBackend(config, ledger, ledger.create(config.model_dump()), client)
    states = initial_state(make_code(config))[None, :]
    gates = HiddenPlant(config).gates(np.linspace(-0.1, 0.1, 18), 0)
    actual = backend.evolve_states(states.copy(), gates, 9, np.random.default_rng(1))
    expected = LocalBackend().evolve_states(states.copy(), gates, 9, np.random.default_rng(1))
    assert np.allclose(actual, expected, atol=1e-12)
    assert client.calls > 0
