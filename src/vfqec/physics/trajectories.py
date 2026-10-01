"""Exact state-vector trajectories with projective stabilizers and finite sampling."""

import numpy as np

from vfqec.decoders.matching import decoder_for
from vfqec.physics.operators import hadamard_all, initial_state, pauli


class TrajectoryEngine:
    """Analytically eliminates ideal extraction ancillas, preserving the data instrument."""

    def __init__(self, code, plant, backend, shots: int):
        self.code, self.plant, self.backend = code, plant, backend
        self.states = np.tile(initial_state(code), (shots, 1))
        self.decoder = decoder_for(code)
        self.shots = shots
        # Final destructive-readout outcomes, decoded without adding a physical measurement.
        indices = np.arange(2**code.n)
        bits = ((indices[:, None] >> np.arange(code.n)) & 1).astype(np.uint8)
        matrix = code.check_matrix(code.basis)
        synd = np.zeros((len(indices), code.m), dtype=np.uint8)
        if code.basis == "Z":
            synd[:, len(code.x_checks) :] = (bits @ matrix.T) % 2
        else:
            synd[:, : len(code.x_checks)] = (bits @ matrix.T) % 2
        cx, cz = self.decoder.decode(synd)
        corrected = bits ^ (cx if code.basis == "Z" else cz)
        support = code.logical_z if code.basis == "Z" else code.logical_x
        self.readout_errors = corrected[:, list(support)].sum(axis=1) % 2

    def step(
        self, theta: np.ndarray, t: int, rng: np.random.Generator, twirl: bool = False
    ) -> tuple[float, int, int]:
        c, n = self.plant.config, self.code.n
        self.states = self.backend.evolve_states(
            self.states, self.plant.gates(theta, t, twirl), n, rng
        )
        for i in range(n):
            event = rng.random(self.shots) < c.p
            kinds = (
                rng.integers(0, 3, self.shots)
                if self.code.name == "surface"
                else np.full(self.shots, 0 if c.basis == "Z" else 2)
            )
            for k, axis in enumerate("XYZ"):
                take = event & (kinds == k)
                if take.any():
                    self.states[take] = pauli(self.states[take], n, axis, (i,))
        observed = np.zeros((self.shots, self.code.m), dtype=np.uint8)
        for j, (axis, support) in enumerate(self.code.checks):
            transformed = pauli(self.states, n, axis, support)
            expectation = np.einsum("ij,ij->i", self.states.conj(), transformed).real
            pminus = np.clip((1 - expectation) / 2, 0, 1)
            outcome = rng.random(self.shots) < pminus
            sign = 1 - 2 * outcome.astype(int)
            self.states += sign[:, None] * transformed
            self.states /= np.linalg.norm(self.states, axis=1)[:, None]
            observed[:, j] = outcome ^ (rng.random(self.shots) < c.q)
        cx, cz = self.decoder.decode(observed)
        for i in range(n):
            for axis, corrections in (("X", cx), ("Z", cz)):
                take = corrections[:, i].astype(bool)
                if take.any():
                    self.states[take] = pauli(self.states[take], n, axis, (i,))
        final = hadamard_all(self.states, n) if c.basis == "X" else self.states
        probabilities = np.einsum("ij,j->i", np.abs(final) ** 2, self.readout_errors)
        # Hypothetical destructive endpoint readout; do not collapse ongoing memory.
        failures = rng.random(self.shots) < np.clip(probabilities, 0, 1)
        return float(failures.mean()), int(observed.sum()), observed.size
