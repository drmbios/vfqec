"""Vectorized Pauli operations without constructing dense Kronecker matrices."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=512)
def pauli_map(n: int, axis: str, support: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(2**n)
    mask = sum(1 << i for i in support)
    perm = idx ^ mask if axis in ("X", "Y") else idx
    parity = np.zeros(2**n, dtype=int)
    for i in support:
        parity ^= (perm >> i) & 1
    phase = (1j ** len(support) if axis == "Y" else 1) * (-1.0) ** parity
    if axis == "X":
        phase = np.ones(2**n)
    return perm, phase


def pauli(states: np.ndarray, n: int, axis: str, support: tuple[int, ...]) -> np.ndarray:
    """Apply a Pauli to rows of state vectors (or one vector)."""
    perm, phase = pauli_map(n, axis, support)
    return states[..., perm] * phase


def rotate(
    states: np.ndarray, n: int, axis: str, support: tuple[int, ...], angle: float | np.ndarray
) -> np.ndarray:
    angle = np.asarray(angle)
    if angle.ndim:
        angle = angle[:, None]
    return np.cos(angle / 2) * states - 1j * np.sin(angle / 2) * pauli(states, n, axis, support)


def density_pauli(rho: np.ndarray, n: int, axis: str, support: tuple[int, ...]) -> np.ndarray:
    perm, phase = pauli_map(n, axis, support)
    return rho[np.ix_(perm, perm)] * phase[:, None] * phase.conj()[None, :]


def density_rotate(
    rho: np.ndarray, n: int, axis: str, support: tuple[int, ...], angle: float
) -> np.ndarray:
    # Columns of rho are ket vectors; rows are conjugate bra vectors.
    left = rotate(rho.T, n, axis, support, angle).T
    return rotate(left.conj(), n, axis, support, angle).conj()


def hadamard_all(states: np.ndarray, n: int) -> np.ndarray:
    result = states.copy()
    for i in range(n):
        result = (pauli(result, n, "X", (i,)) + pauli(result, n, "Z", (i,))) / np.sqrt(2)
    return result


def initial_state(code) -> np.ndarray:
    state = np.zeros(2**code.n, dtype=complex)
    state[0] = 1
    if code.name == "repetition":
        return hadamard_all(state, code.n) if code.basis == "X" else state
    for support in code.x_checks:
        state += pauli(state, code.n, "X", support)
        state /= np.linalg.norm(state)
    if code.basis == "X":
        state += pauli(state, code.n, "X", code.logical_x)
        state /= np.linalg.norm(state)
    return state


def apply_subsystem(
    states: np.ndarray, unitary: np.ndarray, support: tuple[int, ...], n: int
) -> np.ndarray:
    """Apply a small little-endian unitary to selected qubits in row state vectors."""
    indices = np.arange(2**n)
    mask = sum(1 << i for i in support)
    bases = indices[(indices & mask) == 0]
    local = np.arange(2 ** len(support))
    offsets = sum(((local >> j) & 1) << qubit for j, qubit in enumerate(support))
    groups = bases[:, None] | offsets[None, :]
    transformed = np.einsum("...j,ij->...i", states[..., groups], unitary)
    result = states.copy()
    result[..., groups] = transformed
    return result
