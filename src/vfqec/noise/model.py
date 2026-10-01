"""Hidden plant and the explicitly ordered coherent-noise/compensation layer."""

from dataclasses import dataclass

import numpy as np

from vfqec.config import Config


@dataclass(frozen=True)
class Gate:
    axis: str
    support: tuple[int, ...]
    angle: float
    twirl: bool = False


class HiddenPlant:
    """Only execution code and post-experiment diagnostics may inspect this object."""

    def __init__(self, config: Config):
        self.config = config
        self.n = config.n

    def field(self, round_index: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        drift = c.drift_amplitude * np.sin(2 * np.pi * round_index / c.drift_period)
        drift += c.drift_slope * round_index
        x = np.broadcast_to(c.eps_x, (self.n,)).copy()
        z = np.broadcast_to(c.eps_z, (self.n,)).copy()
        if c.code == "surface" or c.basis == "Z":
            x += drift
        if c.code == "surface" or c.basis == "X":
            z += drift
        return x, z

    def oracle_theta(self, round_index: int) -> np.ndarray:
        x, z = self.field(round_index)
        if self.config.code == "surface":
            return np.concatenate([x, z])
        return x if self.config.basis == "Z" else z

    def gates(
        self, theta: np.ndarray, round_index: int, twirl: bool = False, unencoded: bool = False
    ) -> list[Gate]:
        x, z = self.field(round_index)
        n = 1 if unencoded else self.n
        tx, tz = np.zeros(self.n), np.zeros(self.n)
        if not unencoded:
            if self.config.code == "surface":
                tx, tz = theta[: self.n], theta[self.n :]
            elif self.config.basis == "Z":
                tx = theta
            else:
                tz = theta
        gates = []
        for i in range(n):
            # Rightmost inverse first: Rx(eps), Rz(eps), Rz(-theta), Rx(-theta).
            for axis, angle, is_noise in (
                ("X", x[i], True),
                ("Z", z[i], True),
                ("Z", -tz[i], False),
                ("X", -tx[i], False),
            ):
                if angle:
                    gates.append(Gate(axis, (i,), float(angle), twirl and is_noise))
        if not unencoded and self.config.zz:
            edges = [(i, i + 1) for i in range(n - 1)]
            if self.config.code == "surface":
                edges = [
                    (i, j)
                    for i in range(9)
                    for j in range(i + 1, 9)
                    if (j == i + 1 and i // 3 == j // 3) or j == i + 3
                ]
            gates.extend(Gate("Z", edge, self.config.zz, twirl) for edge in edges)
        return gates
