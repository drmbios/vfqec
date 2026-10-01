"""Exact NumPy gate propagation and exact gate-wise Pauli twirling."""

import numpy as np

from vfqec.noise.model import Gate
from vfqec.physics.operators import density_pauli, density_rotate, rotate


class LocalBackend:
    label = "local simulation / NumPy"

    def evolve_density(self, rho: np.ndarray, gates: list[Gate], n: int) -> np.ndarray:
        for gate in gates:
            if gate.twirl:
                prob = np.sin(gate.angle / 2) ** 2
                rho = (1 - prob) * rho + prob * density_pauli(rho, n, gate.axis, gate.support)
            else:
                rho = density_rotate(rho, n, gate.axis, gate.support, gate.angle)
        return rho

    def evolve_states(
        self, states: np.ndarray, gates: list[Gate], n: int, rng: np.random.Generator
    ) -> np.ndarray:
        for gate in gates:
            angle = gate.angle
            if gate.twirl:
                angle *= rng.choice([-1, 1], len(states))
            states = rotate(states, n, gate.axis, gate.support, angle)
        return states
