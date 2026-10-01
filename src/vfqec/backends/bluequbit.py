"""BlueQubit coherent-layer execution with explicit local syndrome instrumentation.

A Choi state returns the coherent unitary using one remote circuit. This avoids assuming
support for reset/mid-circuit feedback. It is simulation, never hardware validation.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from vfqec.noise.model import Gate
from vfqec.physics.circuits import append_gates


class BlueQubitBackend:
    def __init__(self, config, ledger, run_id: str, client=None):
        if client is None:
            import bluequbit

            token = os.getenv("BLUEQUBIT_API_TOKEN")
            if not token:
                raise ValueError("Set BLUEQUBIT_API_TOKEN to use BlueQubit")
            client = bluequbit.init(api_token=token, execution_mode="cloud")
        self.client, self.config, self.ledger, self.run_id = client, config, ledger, run_id
        self.label = (
            f"BlueQubit {config.device} coherent simulation + local noise/syndrome/recovery"
        )
        self.cache = {}
        existing = ledger.run_jobs(run_id)
        self.count = sum(bool(job["job_id"]) for job in existing)
        self.spent = sum(float(job["cost"] or 0) for job in existing)
        self.path = Path(os.getenv("RESULTS_DIR", "results")) / "remote-cache"
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _status(exc: Exception) -> int | None:
        return getattr(exc, "http_status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )

    def _request(self, operation, *, submission: bool = False):
        """Retry rate-limit rejections; never resubmit after an ambiguous transport failure."""
        for attempt in range(5):
            try:
                return operation()
            except Exception as exc:
                status = self._status(exc)
                retryable = status == 429 or (not submission and status in (500, 502, 503, 504))
                if not retryable or attempt == 4:
                    raise
                self.ledger.event(
                    self.run_id,
                    {
                        "type": "rate_limit_retry",
                        "attempt": attempt + 1,
                        "status": status,
                        "submission": submission,
                    },
                )
                time.sleep(min(2**attempt, 16))
        raise RuntimeError("Retry limit exhausted")

    def _unitary(self, gates: list[Gate], n: int) -> np.ndarray:
        key = hashlib.sha256(
            json.dumps(
                {
                    "device": self.config.device,
                    "n": n,
                    "gates": [asdict(g) for g in gates],
                    "version": 1,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        # Per-run key prevents accidental reuse of remote results across billing provenance.
        job_key = f"{self.run_id}:{key}"
        prior = self.ledger.claim_job(job_key, self.run_id, self.label)
        path = self.path / f"{job_key.replace(':', '-')}.npy"
        if prior and prior["status"] == "complete" and path.exists():
            unitary = np.load(path)
            self.cache[key] = unitary
            return unitary
        if prior and not prior["job_id"]:
            raise RuntimeError(
                "An earlier submission has uncertain status; inspect ledger before retry"
            )
        if prior:
            job_id = prior["job_id"]
        else:
            if self.count >= self.config.max_remote_jobs:
                raise RuntimeError("Remote job budget exhausted")
            circuit = QuantumCircuit(2 * n)
            for i in range(n):
                circuit.h(n + i)
                circuit.cx(n + i, i)
            append_gates(circuit, gates)
            # SDK estimate support varies by device. Refuse unknown estimates when capped.
            estimate = self._request(
                lambda: self.client.estimate(circuit, device=self.config.device)
            )
            estimated = getattr(estimate, "estimated_cost", None)
            if estimated is None or not np.isfinite(float(estimated)) or float(estimated) < 0:
                raise RuntimeError(
                    "Backend supplied no credit estimate; refusing unpriced submission"
                )
            if self.spent + float(estimated) > self.config.max_credits:
                raise RuntimeError("Estimated BlueQubit credit budget exceeded")
            # Do not blindly retry an ambiguous submission (could charge twice).
            job = self._request(
                lambda: self.client.run(
                    circuit,
                    device=self.config.device,
                    asynchronous=True,
                    job_name=f"vfqec-{self.run_id[:8]}-{key[:12]}",
                    tags={"vfqec_run": self.run_id, "seed": self.config.seed},
                ),
                submission=True,
            )
            job_id = job.job_id
            self.count += 1
            self.spent += float(estimated)
            self.ledger.save_job(job_key, job_id=job_id, status="submitted", cost=float(estimated))
            self.ledger.event(
                self.run_id,
                {"type": "remote_job", "job_id": job_id, "estimated_cost": float(estimated)},
            )
        for attempt in range(5):
            try:
                result = self.client.wait(job_id, timeout=45)
                break
            except Exception as exc:
                status = self._status(exc)
                if attempt == 4 or status in (400, 401, 403, 404):
                    raise
                self.ledger.event(
                    self.run_id,
                    {
                        "type": "remote_retry",
                        "job_id": job_id,
                        "attempt": attempt + 1,
                        "status": status,
                    },
                )
                time.sleep(min(2**attempt, 16))
        if not result.ok:
            self.ledger.save_job(job_key, status="failed")
            raise RuntimeError(f"BlueQubit job {job_id} failed; inspect provider job details")
        state = np.asarray(result.get_statevector())
        unitary = state.reshape(2**n, 2**n).T * np.sqrt(2**n)
        if not np.isfinite(unitary).all() or not np.allclose(
            unitary.conj().T @ unitary, np.eye(2**n), atol=1e-5
        ):
            raise RuntimeError(
                "Remote Choi state is not unitary; MPS truncation may be too aggressive"
            )
        np.save(path, unitary)
        actual = getattr(result, "cost", None)
        prior_cost = next(
            (
                float(row["cost"] or 0)
                for row in self.ledger.run_jobs(self.run_id)
                if row["key"] == job_key
            ),
            0.0,
        )
        actual = float(actual) if actual is not None and np.isfinite(float(actual)) else prior_cost
        self.spent += actual - prior_cost
        self.ledger.save_job(
            job_key,
            status="complete",
            cost=actual,
            payload=json.dumps({"matrix_path": str(path), "provider_seed": None}),
        )
        self.cache[key] = unitary
        return unitary

    def evolve_density(self, rho: np.ndarray, gates: list[Gate], n: int) -> np.ndarray:
        if not gates:
            return rho
        if any(g.twirl for g in gates):
            for gate in gates:
                u = self._unitary([replace(gate, twirl=False)], n)
                evolved = u @ rho @ u.conj().T
                rho = (evolved + u.conj().T @ rho @ u) / 2 if gate.twirl else evolved
            return rho
        u = self._unitary(gates, n)
        return u @ rho @ u.conj().T

    def evolve_states(
        self, states: np.ndarray, gates: list[Gate], n: int, rng: np.random.Generator
    ) -> np.ndarray:
        if not gates:
            return states
        if any(g.twirl for g in gates):
            for gate in gates:
                u = self._unitary([replace(gate, twirl=False)], n)
                take = rng.random(len(states)) < 0.5 if gate.twirl else np.ones(len(states), bool)
                states[take] = states[take] @ u.T
                states[~take] = states[~take] @ u.conj()
            return states
        return states @ self._unitary(gates, n).T
