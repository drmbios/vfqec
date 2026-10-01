"""Confidence intervals and conservative fits; no invented thresholds or rates."""

import json

import numpy as np
from scipy.optimize import curve_fit


def wilson(errors, shots: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(errors, dtype=float) / shots
    z = 1.959963984540054
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    half = z / denominator * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots))
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def decay(rounds, rate):
    return (1 - np.exp(-2 * rate * rounds)) / 2


def analyze(result: dict) -> dict:
    fits, anomalies, reruns = {}, [], []
    for method, values in result["methods"].items():
        y = np.asarray(values["logical_error"])
        x = np.arange(1, len(y) + 1)
        lo, hi = wilson(values["errors"], values["shots"])
        if not np.any(values["errors"]) or len(y) < 3:
            fit = {"rate": None, "status": "insufficient observed failures", "rmse": None}
        else:
            rate, _ = curve_fit(decay, x, y, p0=[0.001], bounds=(0, 5), maxfev=10000)
            rmse = float(np.sqrt(np.mean((decay(x, rate[0]) - y) ** 2)))
            fit = {
                "rate": float(rate[0]),
                "status": "phenomenological exponential fit",
                "rmse": rmse,
            }
            if rmse > max(0.02, 3 / np.sqrt(values["shots"])):
                anomalies.append(
                    f"{method}: exponential model poorly describes coherent/drifting memory"
                )
                reruns.append({"method": method, "reason": "poor decay fit", "shot_multiplier": 2})
        fit.update(final_error=float(y[-1]), final_ci95=[float(lo[-1]), float(hi[-1])])
        fits[method] = fit
    history = result.get("optimizer", [])
    starts = sorted({row["calibration_round"] for row in history})
    for start in starts:
        rows = [
            r
            for r in history
            if r["calibration_round"] == start and r["phase"] in ("initial", "candidate")
        ]
        if len(rows) > 1 and np.mean([r["cost"] for r in rows[-5:]]) >= rows[0]["cost"]:
            anomalies.append(f"Calibration at round {start}: no measured convergence")
            reruns.append(
                {"calibration_round": start, "reason": "non-convergence", "shot_multiplier": 2}
            )
    return {
        "fits": fits,
        "anomalies": anomalies,
        "reruns": reruns,
        "fit_caution": "Rates are descriptive; coherent/drifting noise need not be exponential. "
        "No fit covariance is interpreted as independent-round uncertainty.",
        "rerun_policy": "Requests are recorded, not automatically billed or run beyond the budget.",
    }


def analyze_sweep(results: list[dict]) -> dict:
    """Fit distance suppression and bracketed finite-size crossings."""

    def comparable(c):
        keys = (
            "code",
            "basis",
            "q",
            "eps_x",
            "eps_z",
            "zz",
            "drift_amplitude",
            "drift_period",
            "drift_slope",
            "rounds",
            "cadence",
            "optimizer",
            "iterations",
            "shots",
            "window",
            "backend",
            "device",
            "aer_method",
        )
        values = {key: c[key] for key in keys}
        for field in ("eps_x", "eps_z"):
            if len(set(values[field])) == 1:
                values[field] = values[field][:1]
        return json.dumps(values, sort_keys=True)

    signatures = {comparable(r["config"]) for r in results}
    groups = {}
    for r in results:
        c = r["config"]
        for method, values in r["methods"].items():
            key = (c["p"], c["q"], c["rounds"], c["basis"], c["code"], method, comparable(c))
            groups.setdefault(key, []).append((c["distance"], values["logical_error"][-1]))
    suppression = []
    for key, pairs in groups.items():
        pairs = sorted(set(pairs))
        for (d1, p1), (d2, p2) in zip(pairs, pairs[1:]):
            suppression.append(
                {
                    "p": key[0],
                    "q": key[1],
                    "rounds": key[2],
                    "basis": key[3],
                    "code": key[4],
                    "method": key[5],
                    "distances": [d1, d2],
                    "factor_per_two_distance": (p1 / p2) ** (2 / (d2 - d1))
                    if p1 > 0 and p2 > 0 and d2 != d1
                    else None,
                }
            )
    thresholds = {}
    for method in results[0]["methods"] if results else []:
        if len(signatures) != 1:
            thresholds[method] = {"status": "mixed configurations; no joint threshold fit"}
            continue
        rows = [
            (r["config"]["p"], r["config"]["distance"], r["methods"][method]["logical_error"][-1])
            for r in results
        ]
        distances = sorted({d for _, d, _ in rows})
        ps = sorted({p for p, _, _ in rows})
        if len(distances) < 3 or len(ps) < 3:
            thresholds[method] = {"status": "insufficient distances/noise points"}
            continue
        differences = []
        for p in ps:
            at_p = {d: y for pp, d, y in rows if pp == p}
            if distances[0] in at_p and distances[-1] in at_p:
                differences.append(at_p[distances[-1]] - at_p[distances[0]])
        if not differences or min(differences) >= 0 or max(differences) <= 0:
            thresholds[method] = {"status": "no crossing bracketed; no threshold claimed"}
            continue

        def scaling(x, a, b, pc, nu):
            p, d = x
            return a + b * (p - pc) * d ** (1 / nu)

        p, d, y = np.asarray(rows).T
        try:
            pars, _ = curve_fit(
                scaling,
                (p, d),
                y,
                p0=[0.1, 1, np.median(ps), 1],
                bounds=([0, 0, min(ps), 0.2], [1, 100, max(ps), 10]),
                maxfev=20000,
            )
            thresholds[method] = {
                "status": "exploratory finite-size crossing fit",
                "p_crossing": float(pars[2]),
                "nu": float(pars[3]),
            }
        except (ValueError, RuntimeError):
            thresholds[method] = {"status": "fit failed"}
    return {
        "memory_tradeoff": [
            {
                "run_id": r["run_id"],
                "backend": r["backend"],
                "cadence": r["config"]["cadence"],
                "shots_per_evaluation": r["config"]["shots"],
                "calibration_shots": sum(update["shots"] for update in r["updates"]),
                "method": method,
                "final_logical_error": values["logical_error"][-1],
            }
            for r in results
            for method, values in r["methods"].items()
        ],
        "suppression": suppression,
        "thresholds": thresholds,
        "caution": "Fixed-round memory crossings are not circuit-level thresholds.",
    }
