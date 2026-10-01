"""CSS code definitions; data bit zero is Qiskit's least significant qubit."""

from dataclasses import dataclass

import numpy as np

from vfqec.config import Config


@dataclass(frozen=True)
class Code:
    """Stabilizer supports and logical observables on data qubits only."""

    name: str
    n: int
    x_checks: tuple[tuple[int, ...], ...]
    z_checks: tuple[tuple[int, ...], ...]
    logical_x: tuple[int, ...]
    logical_z: tuple[int, ...]
    basis: str = "Z"

    @property
    def checks(self) -> tuple:
        return tuple(("X", s) for s in self.x_checks) + tuple(("Z", s) for s in self.z_checks)

    @property
    def m(self) -> int:
        return len(self.checks)

    def check_matrix(self, axis: str) -> np.ndarray:
        supports = self.x_checks if axis == "X" else self.z_checks
        matrix = np.zeros((len(supports), self.n), dtype=np.uint8)
        for row, support in enumerate(supports):
            matrix[row, list(support)] = 1
        return matrix


def repetition(n: int = 3, basis: str = "Z") -> Code:
    """Even lengths are allowed, with a fixed minimum-weight tie convention."""
    if not 3 <= n <= 9 or basis not in ("X", "Z"):
        raise ValueError("Repetition requires 3 <= N <= 9 and basis X or Z")
    checks = tuple((i, i + 1) for i in range(n - 1))
    return Code(
        "repetition",
        n,
        checks if basis == "X" else (),
        checks if basis == "Z" else (),
        tuple(range(n)) if basis == "Z" else (0,),
        (0,) if basis == "Z" else tuple(range(n)),
        basis,
    )


def surface(basis: str = "Z") -> Code:
    """Rotated [[9,1,3]] patch: nine data and eight extraction ancillas (17 total)."""
    return Code(
        "surface",
        9,
        ((0, 1, 3, 4), (2, 5), (3, 6), (4, 5, 7, 8)),
        ((0, 1), (1, 2, 4, 5), (3, 4, 6, 7), (7, 8)),
        (0, 1, 2),
        (0, 3, 6),
        basis,
    )


def make_code(config: Config) -> Code:
    return (
        repetition(config.distance, config.basis)
        if config.code == "repetition"
        else surface(config.basis)
    )
