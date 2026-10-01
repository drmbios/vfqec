"""Aer density-matrix propagation and independent Clifford circuit execution."""

import os
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from vfqec.backends.local import LocalBackend
from vfqec.noise.model import Gate
from vfqec.physics.circuits import append_gates, coherent_circuit


class AerBackend(LocalBackend):
    def __init__(self, method: str = "density_matrix", seed: int = 0):
        self.method = method
        self.seed = seed
        self.device = os.getenv("AER_DEVICE", "CPU")
        self.label = f"local simulation / Qiskit Aer {method} ({self.device})"
        self.simulator = AerSimulator(method=method, device=self.device, max_parallel_threads=1)

    def evolve_density(self, rho: np.ndarray, gates: list[Gate], n: int) -> np.ndarray:
        if self.method == "stabilizer":
            raise ValueError(
                "Arbitrary coherent-field experiments are not Clifford; use density_matrix"
            )
        # Twirled gates are exact stochastic Pauli channels, delegated to the shared channel engine.
        if any(g.twirl for g in gates):
            return super().evolve_density(rho, gates, n)
        circuit = QuantumCircuit(n)
        circuit.set_density_matrix(rho)
        append_gates(circuit, gates)
        circuit.save_density_matrix()
        sim = (
            self.simulator
            if self.method == "density_matrix"
            else AerSimulator(method="density_matrix", device=self.device, max_parallel_threads=1)
        )
        return np.asarray(
            sim.run(circuit, seed_simulator=self.seed).result().data(0)["density_matrix"]
        )

    def evolve_states(
        self, states: np.ndarray, gates: list[Gate], n: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.method == "stabilizer":
            raise ValueError("Use run_counts for Clifford-only circuits")
        if any(g.twirl for g in gates):
            # Gate-wise sign twirling is a shared exact ensemble operation.
            return super().evolve_states(states, gates, n, rng)
        return states @ self._unitary(tuple(gates), n).T

    @lru_cache(maxsize=32)
    def _unitary(self, gates: tuple[Gate, ...], n: int) -> np.ndarray:
        circuit = coherent_circuit(n, gates)
        circuit.save_unitary()
        sim = AerSimulator(method="unitary", device=self.device, max_parallel_threads=1)
        return np.asarray(sim.run(circuit).result().data(0)["unitary"])

    def run_counts(self, circuit, shots: int = 1024) -> dict:
        """Supports stabilizer mode for Clifford circuits; Aer rejects non-Clifford gates."""
        return (
            self.simulator.run(circuit, shots=shots, seed_simulator=self.seed).result().get_counts()
        )
