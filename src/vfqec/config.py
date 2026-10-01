"""Validated, reproducible experiment configurations and resource limits."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    """Private plant configuration. Never supplied to the field optimizer or an LLM."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    experiment: str = "E1"
    code: Literal["repetition", "surface"] = "repetition"
    distance: int = Field(3, ge=3, le=9)
    basis: Literal["X", "Z"] = "Z"
    p: float = Field(0.002, ge=0, le=0.5)
    q: float = Field(0.0, ge=0, lt=0.5)
    eps_x: list[float] = Field(default_factory=lambda: [0.12, -0.08, 0.15])
    eps_z: list[float] = Field(default_factory=lambda: [0.0])
    zz: float = 0.0
    drift_amplitude: float = 0.0
    drift_period: float = Field(60.0, gt=0)
    drift_slope: float = 0.0
    rounds: int = Field(60, ge=1, le=1000)
    shots: int = Field(4000, ge=1, le=100000)
    evaluation_shots: int = Field(4000, ge=1, le=100000)
    window: int = Field(2, ge=1, le=20)
    iterations: int = Field(50, ge=1, le=1000)
    cadence: int = Field(20, ge=1)
    optimizer: Literal["spsa", "bayes"] = "spsa"
    seed: int = Field(1729, ge=0, le=2**31 - 1)
    backend: Literal["local", "aer", "bluequbit"] = "local"
    device: Literal["cpu", "gpu", "mps.cpu", "mps.gpu"] = "cpu"
    aer_method: Literal["density_matrix", "statevector", "stabilizer"] = "density_matrix"
    max_shots: int = Field(20_000_000, ge=1)
    max_remote_jobs: int = Field(1000, ge=0)
    max_credits: float = Field(10.0, ge=0)
    batch_size: int = Field(256, ge=1, le=4096)

    @model_validator(mode="after")
    def validate_physics(self) -> "Config":
        if self.code == "surface" and self.distance != 3:
            raise ValueError("Only rotated surface d=3 is implemented; d=5 is optional and absent")
        for field in (self.eps_x, self.eps_z):
            if len(field) not in (1, self.n):
                raise ValueError(f"Fields must have length 1 or {self.n}")
        if self.code == "repetition":
            invisible = self.eps_z if self.basis == "Z" else self.eps_x
            if any(invisible) or self.zz != 0:
                raise ValueError(
                    "Repetition memory supports its protected axis only; use surface "
                    "code to investigate simultaneous X/Z fields or ZZ crosstalk"
                )
        return self

    @property
    def n(self) -> int:
        return self.distance if self.code == "repetition" else self.distance**2

    @property
    def dimensions(self) -> int:
        return self.n if self.code == "repetition" else 2 * self.n

    @property
    def calibrations(self) -> int:
        return 1 + (self.rounds - 1) // self.cadence

    @property
    def shot_bound(self) -> int:
        # SPSA: initial + (plus, minus, candidate) per iteration. Both learned methods
        # share initial calibration. Evaluation shots count circuit trajectories.
        evaluations = 1 + (3 if self.optimizer == "spsa" else 1) * self.iterations
        return (
            self.calibrations * evaluations * self.shots + 6 * self.evaluation_shots * self.rounds
        )


def preset(name: str, **overrides: object) -> Config:
    """Return one run; CLI expands E3 and E4 into sweeps."""
    base: dict = {"experiment": name}
    if name == "E2":
        base.update(
            code="surface",
            distance=3,
            eps_x=[0.12],
            eps_z=[-0.08],
            zz=0.04,
            rounds=20,
            shots=512,
            evaluation_shots=512,
            iterations=30,
            max_remote_jobs=4000,
        )
    elif name == "E3":
        base.update(eps_x=[0.12], rounds=30, shots=1000, evaluation_shots=2000)
    elif name == "E4":
        base.update(drift_amplitude=0.10, drift_slope=0.0005, cadence=10)
    elif name == "E5":
        raise ValueError("E5 is disabled; use 'vfqec hardware-template' to export the template")
    elif name != "E1":
        raise ValueError(f"Unknown experiment {name}")
    base.update(overrides)
    return Config.model_validate(base)
