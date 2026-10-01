"""A backend applies coherent layers; shared simulators implement the QEC instrument."""

from typing import Protocol

import numpy as np

from vfqec.noise.model import Gate


class Backend(Protocol):
    label: str

    def evolve_density(self, rho: np.ndarray, gates: list[Gate], n: int) -> np.ndarray: ...

    def evolve_states(
        self, states: np.ndarray, gates: list[Gate], n: int, rng: np.random.Generator
    ) -> np.ndarray: ...
