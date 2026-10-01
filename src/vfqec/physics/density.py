"""Exact small-repetition density matrix channel, including noisy syndrome recovery."""

import numpy as np

from vfqec.decoders.lookup import lookup
from vfqec.physics.operators import density_pauli, hadamard_all, initial_state


class DensityEngine:
    """Integrate physical and measurement noise exactly; sample only observations."""

    def __init__(self, code, plant, backend, unencoded: bool = False):
        self.code, self.plant, self.backend = code, plant, backend
        self.unencoded = unencoded
        self.n = 1 if unencoded else code.n
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1
        if unencoded and code.basis == "X":
            state = hadamard_all(state, self.n)
        elif not unencoded:
            state = initial_state(code)
        self.rho = np.outer(state, state.conj())
        self.indices = np.arange(2**self.n)
        self.m = 0 if unencoded else self.n - 1
        self.syndromes = (
            sum((((self.indices >> i) ^ (self.indices >> (i + 1))) & 1) << i for i in range(self.m))
            if self.m
            else np.zeros(2, dtype=int)
        )
        self.weights = np.array([s.bit_count() for s in range(2**self.m)])
        self.table = lookup(self.n) if self.m else np.array([0])
        self.readout_errors = np.array([self._logical_bit(i) for i in self.indices])
        q = plant.config.q
        self.confusion = np.array(
            [
                [
                    (q ** (s ^ o).bit_count()) * (1 - q) ** (self.m - (s ^ o).bit_count())
                    for o in range(2**self.m)
                ]
                for s in range(2**self.m)
            ]
        )

    def _logical_bit(self, index: int) -> int:
        if self.unencoded:
            return index
        # Decode final ideal data readout. Odd N equals majority; even N uses LUT tie break.
        syndrome = sum((((index >> i) ^ (index >> (i + 1))) & 1) << i for i in range(self.m))
        return int((index ^ self.table[syndrome]) & 1)

    def _readout_basis(self, rho: np.ndarray) -> np.ndarray:
        if self.code.basis == "Z":
            return rho
        return hadamard_all(hadamard_all(rho.T, self.n).T, self.n)

    def step(
        self, theta: np.ndarray, t: int, rng: np.random.Generator, shots: int, twirl: bool = False
    ) -> tuple[float, int, int]:
        gates = self.plant.gates(theta, t, twirl, self.unencoded)
        self.rho = self.backend.evolve_density(self.rho, gates, self.n)
        p = self.plant.config.p
        axis = "X" if self.code.basis == "Z" else "Z"
        if self.unencoded and self.code.name == "surface":
            for i in range(self.n):
                self.rho = (1 - p) * self.rho + p / 3 * sum(
                    density_pauli(self.rho, self.n, a, (i,)) for a in "XYZ"
                )
        else:
            for i in range(self.n):
                self.rho = (1 - p) * self.rho + p * density_pauli(self.rho, self.n, axis, (i,))
        rho = self._readout_basis(self.rho)
        fired = 0
        if self.m:
            probs = (
                np.bincount(self.syndromes, weights=rho.diagonal().real, minlength=2**self.m)
                @ self.confusion
            )
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            fired = int(rng.multinomial(shots, probs) @ self.weights)
            corrected = np.zeros_like(rho)
            for syndrome in range(2**self.m):
                idx = self.indices[self.syndromes == syndrome]
                block = rho[np.ix_(idx, idx)]
                for observed, probability in enumerate(self.confusion[syndrome]):
                    if probability:
                        out = idx ^ self.table[observed]
                        corrected[np.ix_(out, out)] += probability * block
            rho = corrected
            self.rho = self._readout_basis(rho)
        probability = float(np.clip(rho.diagonal().real @ self.readout_errors, 0, 1))
        return probability, fired, shots * self.m
