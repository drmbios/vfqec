"""Seed-level analysis and a standalone report for the held-out calibration study."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

PRIMARY_GROUP = "heldout_static"
PRIMARY_SEEDS = tuple(range(2001, 2025))
NONINFERIORITY_MARGIN = 0.001
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20261025
LABELS = {
    "unencoded": "Unencoded",
    "standard": "Standard QEC",
    "twirled": "QEC + twirling",
    "oracle": "Oracle",
    "spsa_budget": "SPSA · 61k shots",
    "spsa_reference": "SPSA · 604k shots",
    "fourier": "Fourier · 60k shots",
    "adaptive_spsa": "Adaptive SPSA",
    "adaptive_fourier": "Adaptive Fourier",
}
PRIOR_ART = [
    {
        "title": "Ostaszewski et al. — Structure optimization for parameterized quantum circuits",
        "url": "https://arxiv.org/abs/1905.09692",
        "relevance": "Sinusoidal coordinate minimization (Rotosolve) predates this software.",
    },
    {
        "title": "Wagner et al. — Optimal noise estimation from syndrome statistics",
        "url": "https://arxiv.org/abs/2010.02243",
        "relevance": "Noise estimation and identifiability from syndrome data are established.",
    },
    {
        "title": "Magann et al. — Fast-feedback protocols for calibration and drift control",
        "url": "https://arxiv.org/abs/2512.07815",
        "relevance": "Prior work includes simulated syndrome-only calibration during QEC.",
    },
    {
        "title": "Kunjummen and Taylor — In situ calibration during quantum error correction",
        "url": "https://arxiv.org/abs/2511.01080",
        "relevance": "Prior work combines decoder feedback and injected calibration operations.",
    },
    {
        "title": "Özgüler — Securing quantum error correction against misleading advice",
        "url": "https://arxiv.org/abs/2609.19090",
        "relevance": (
            "Recovery validation requires attention to uncertainty, drift, and sign ambiguity."
        ),
    },
    {
        "title": "Pan et al. — Identifying the Sign of Coherent Over-Rotations",
        "url": "https://arxiv.org/abs/2607.21663",
        "relevance": (
            "Passive observations need not identify coherent signs; active probes can help."
        ),
    },
]
LIMITATIONS = [
    "This is a model-specific exploratory software study with held-out data, not a universal "
    "performance claim, hardware experiment, or fault-tolerant threshold measurement.",
    "The primary endpoint is the observed final-round failure probability; exact probabilities "
    "are secondary evaluation-only diagnostics and are not used to select a controller.",
    "The 10,000-repeat percentile bootstrap resamples paired independent cases, never correlated "
    "round endpoints or stabilizer bits. Its interval describes this sampled case distribution.",
    "The predeclared noninferiority margin is 0.001 absolute failure probability (0.1 percentage "
    "points). This study-specific tolerance does not mean equal performance.",
    "Equal-budget calibration is approximate: 60,000 versus 61,000 probe shots. All fitting and "
    "validation probes count toward each optimizer's reported budget.",
    "A lower syndrome cost does not certify a lower logical error. The three-point sinusoidal "
    "coordinate identity applies to a one-round probe with one occurrence of each control.",
    "State preparation, extraction, recovery, and terminal data readout are otherwise ideal; "
    "the decoder is spatial per round. Calibration uses fresh probes and freezes memory time.",
    "Zero observed failures do not establish zero physical error. Extension groups and exact "
    "probability diagnostics are exploratory, without multiplicity-adjusted confirmation.",
    "BlueQubit verification runs reuse local design cases and are not additional independent "
    "primary observations. The cloud backend supplies coherent simulation; noise, syndrome "
    "measurement, and recovery are local instrumentation.",
    "This adapts known Rotosolve and syndrome-calibration ideas. Scientific novelty is not "
    "established by a successful benchmark or by this literature review.",
    "The frozen drift runner executed three reference-SPSA calibrations but evaluated only "
    "its static reference arm. Later reference recalibrations were unused by that arm; their "
    "probe shots remain included in total executed calibration costs and in raw records. "
    "Static and adaptive arms can share calibrations, so do not sum arm costs as independent work.",
]


def _is_cloud(result: dict) -> bool:
    return bool(result.get("remote_jobs")) or "bluequbit" in result.get("backend", "").lower()


def _method_summary(values: dict) -> dict:
    errors = np.asarray(values["errors"], dtype=float)
    probabilities = np.asarray(values["logical_error"], dtype=float)
    shots = values["shots"]
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("Evaluation shots must be a positive integer")
    if (
        errors.ndim != 1
        or len(errors) == 0
        or probabilities.shape != errors.shape
        or not np.isfinite(errors).all()
        or not np.equal(errors, np.floor(errors)).all()
        or np.any(errors < 0)
        or np.any(errors > shots)
        or not np.allclose(probabilities, errors / shots, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid logical failure series or shot accounting")
    exact = values.get("exact_probability")
    if exact is not None:
        exact = np.asarray(exact, dtype=float)
        if (
            exact.shape != errors.shape
            or not np.isfinite(exact).all()
            or np.any(exact < 0)
            or np.any(exact > 1)
        ):
            raise ValueError("Invalid evaluation-only probability series")
    return {
        "final_errors": int(errors[-1]),
        "shots": shots,
        "final_error": float(probabilities[-1]),
        "rounds": len(errors),
        "final_exact_probability": float(exact[-1]) if exact is not None else None,
        "engine": values.get("engine"),
        "seed": values.get("seed"),
    }


def _fit_summary(fit: dict) -> dict:
    shots = fit["shots_used"]
    if not isinstance(shots, int) or shots < 0:
        raise ValueError("Calibration shots must be a nonnegative integer")
    return {
        "shots_used": shots,
        "probe_evaluations": len(fit.get("history", [])),
        "theta": fit.get("theta"),
    }


def _calibration_summary(result: dict) -> dict:
    summaries = {}
    for name in sorted(set(result.get("fits", {})) | set(result.get("updates", {}))):
        initial = result.get("fits", {}).get(name)
        updates = result.get("updates", {}).get(name, [])
        if not updates and initial is not None:
            updates = [initial | {"round": 0}]
        costs = [_fit_summary(update) for update in updates]
        if costs and initial is not None and costs[0]["shots_used"] != initial["shots_used"]:
            raise ValueError("Initial fit and first calibration update shot counts disagree")
        unused = costs[1:] if name == "spsa_reference" else []
        summaries[name] = {
            "initial_shots": costs[0]["shots_used"] if costs else 0,
            "executed_shots": sum(c["shots_used"] for c in costs),
            "executed_probe_evaluations": sum(c["probe_evaluations"] for c in costs),
            "calibrations": len(costs),
            "rounds": [u.get("round", 0) for u in updates],
            "unused_reference_recalibration_shots": sum(c["shots_used"] for c in unused),
        }
    return summaries


def _mean_interval(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"mean": None, "ci95": None, "status": "no observations"}
    mean = float(np.mean(values))
    if len(values) == 1:
        return {"mean": mean, "ci95": None, "status": "one case; interval unavailable"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    boot = np.mean(values[indices], axis=1)
    return {
        "mean": mean,
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "status": "paired case percentile bootstrap",
    }


def _comparison(rows: list[dict], reference: str, *, complete: bool) -> dict:
    paired = [r for r in rows if "fourier" in r["methods"] and reference in r["methods"]]
    differences = np.array(
        [
            r["methods"]["fourier"]["final_error"] - r["methods"][reference]["final_error"]
            for r in paired
        ]
    )
    estimate = _mean_interval(differences)
    interval = estimate["ci95"]
    reportable = complete and len(paired) == len(PRIMARY_SEEDS)
    conclusion = "incomplete held-out panel; no primary conclusion"
    if reportable and interval is not None:
        if interval[1] < 0:
            conclusion = "observed mean improvement supported within this model and panel"
        elif interval[1] < NONINFERIORITY_MARGIN:
            conclusion = "noninferiority supported within the predeclared model-specific margin"
        else:
            conclusion = "noninferiority not established at the predeclared margin"
    exact_pairs = [
        r
        for r in paired
        if r["methods"]["fourier"]["final_exact_probability"] is not None
        and r["methods"][reference]["final_exact_probability"] is not None
    ]
    exact_differences = np.array(
        [
            r["methods"]["fourier"]["final_exact_probability"]
            - r["methods"][reference]["final_exact_probability"]
            for r in exact_pairs
        ]
    )
    return {
        "candidate": "fourier",
        "reference": reference,
        "difference_direction": "Fourier minus reference; negative favors Fourier",
        "cases": len(paired),
        "case_ids": [r["id"] for r in paired],
        "mean_difference": estimate["mean"],
        "ci95": interval,
        "interval_status": estimate["status"],
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "conclusions_reportable": reportable,
        "noninferiority_established": bool(
            reportable and interval is not None and interval[1] < NONINFERIORITY_MARGIN
        ),
        "superiority_supported": bool(reportable and interval is not None and interval[1] < 0),
        "conclusion": conclusion,
        "paired_differences": [
            {"id": r["id"], "seed": r["seed"], "difference": float(d)}
            for r, d in zip(paired, differences)
        ],
        "wins_ties_losses": {
            "wins": int(np.sum(differences < 0)),
            "ties": int(np.sum(differences == 0)),
            "losses": int(np.sum(differences > 0)),
        },
        "secondary_exact_probability": {
            "evaluation_only": True,
            "cases": len(exact_pairs),
            **_mean_interval(exact_differences),
        },
    }


def analyze_campaign(results: list[dict]) -> dict:
    """Retain every case while excluding cloud reproductions from primary inference."""
    rows, identities = [], set()
    for result in results:
        case = result["case"]
        cloud = _is_cloud(result)
        identity = (case["id"], cloud)
        if identity in identities:
            raise ValueError("Duplicate case within the same execution population")
        identities.add(identity)
        row = {
            "id": case["id"],
            "group": case["group"],
            "analysis_group": case["group"] + (" / cloud verification" if cloud else ""),
            "seed": int(case["seed"]),
            "config": case["config"],
            "backend": result["backend"],
            "cloud_verification": cloud,
            "commit": result.get("commit"),
            "run_id": result.get("run_id"),
            "remote_jobs": len(result.get("remote_jobs", [])),
            "fits": {k: _fit_summary(v) for k, v in result.get("fits", {}).items()},
            "calibration": _calibration_summary(result),
            "methods": {k: _method_summary(v) for k, v in result["methods"].items()},
        }
        if not row["methods"]:
            raise ValueError("A completed case must contain evaluation methods")
        rows.append(row)
    rows.sort(key=lambda r: (r["analysis_group"], r["seed"], r["id"]))
    primary = [r for r in rows if r["group"] == PRIMARY_GROUP and not r["cloud_verification"]]
    seeds = [r["seed"] for r in primary]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Primary cases must have distinct independent seeds")
    complete = set(seeds) == set(PRIMARY_SEEDS)
    groups = {}
    for name in sorted({r["analysis_group"] for r in rows}):
        members = [r for r in rows if r["analysis_group"] == name]
        methods = {}
        for method in sorted({m for r in members for m in r["methods"]}):
            values = [r["methods"][method] for r in members if method in r["methods"]]
            costs = [r["fits"][method] for r in members if method in r["fits"]]
            finals = np.array([v["final_error"] for v in values])
            methods[method] = {
                "cases": len(values),
                "mean_final_error": float(np.mean(finals)),
                "median_final_error": float(np.median(finals)),
                "min_final_error": float(np.min(finals)),
                "max_final_error": float(np.max(finals)),
                "total_final_errors": sum(v["final_errors"] for v in values),
                "total_evaluation_shots": sum(v["shots"] for v in values),
                "zero_failure_cases": sum(v["final_errors"] == 0 for v in values),
                "calibration_cases": len(costs),
                "total_initial_calibration_shots": sum(v["shots_used"] for v in costs),
                "total_initial_probe_evaluations": sum(v["probe_evaluations"] for v in costs),
            }
        groups[name] = {
            "cases": len(members),
            "case_ids": [r["id"] for r in members],
            "methods": methods,
            "calibration": _aggregate_calibrations(members),
        }
    totals = {
        method: sum(r["fits"].get(method, {}).get("shots_used", 0) for r in primary)
        for method in ("fourier", "spsa_budget", "spsa_reference")
    }
    denominator = totals["fourier"]
    return {
        "schema_version": 1,
        "study": "VFQEC bounded Fourier calibration: model-specific held-out software study",
        "cases": len(rows),
        "primary": {
            "group": PRIMARY_GROUP,
            "expected_cases": len(PRIMARY_SEEDS),
            "expected_seeds": list(PRIMARY_SEEDS),
            "cases": len(primary),
            "observed_seeds": seeds,
            "missing_seeds": sorted(set(PRIMARY_SEEDS) - set(seeds)),
            "unexpected_seeds": sorted(set(seeds) - set(PRIMARY_SEEDS)),
            "complete": complete,
            "endpoint": "observed final-round logical failure probability",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "paired independent case/seed",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "comparisons": {
                "fourier_vs_" + method: _comparison(primary, method, complete=complete)
                for method in ("spsa_reference", "spsa_budget")
            },
            "initial_calibration_shots": totals,
            "initial_shot_ratios_reference_to_fourier": {
                method: totals[method] / denominator if denominator else None
                for method in ("spsa_reference", "spsa_budget")
            },
            "excluded_cloud_case_ids": [
                r["id"] for r in rows if r["group"] == PRIMARY_GROUP and r["cloud_verification"]
            ],
        },
        "groups": groups,
        "calibration": _aggregate_calibrations(rows),
        "case_results": rows,
        "limitations": LIMITATIONS,
        "prior_art": PRIOR_ART,
    }


def _aggregate_calibrations(rows: list[dict]) -> dict:
    totals = {}
    for optimizer in sorted({m for r in rows for m in r["calibration"]}):
        costs = [r["calibration"][optimizer] for r in rows if optimizer in r["calibration"]]
        totals[optimizer] = {
            "cases": len(costs),
            **{
                key: sum(c[key] for c in costs)
                for key in (
                    "initial_shots",
                    "executed_shots",
                    "executed_probe_evaluations",
                    "calibrations",
                    "unused_reference_recalibration_shots",
                )
            },
        }
    return totals


def _figure(summary: dict, path: Path) -> None:
    primary = [
        r
        for r in summary["case_results"]
        if r["group"] == PRIMARY_GROUP and not r["cloud_verification"]
    ]
    colors = {"spsa_budget": "#d48255", "spsa_reference": "#647eb1", "fourier": "#087b61"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    positions = np.arange(len(primary))
    for method, color in colors.items():
        y = [r["methods"].get(method, {}).get("final_error", np.nan) for r in primary]
        axes[0].plot(positions, y, "o-", color=color, label=LABELS[method], linewidth=1.1)
    axes[0].set(title="Held-out cases: observed final-round error", ylabel="Failure probability")
    axes[0].set_xticks(positions, [str(r["seed"]) for r in primary], rotation=45)
    axes[0].set_xlabel("Independent case seed (all observed cases shown)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    for method in ("spsa_budget", "spsa_reference"):
        values = summary["primary"]["comparisons"]["fourier_vs_" + method]["paired_differences"]
        by_id = {r["id"]: r["difference"] for r in values}
        axes[1].plot(
            positions,
            [by_id.get(r["id"], np.nan) for r in primary],
            "o-",
            color=colors[method],
            label="Fourier minus " + LABELS[method],
        )
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].axhline(
        NONINFERIORITY_MARGIN,
        color="#b45139",
        linestyle="--",
        linewidth=1,
        label="Study margin: +0.1 percentage points",
    )
    axes[1].set(
        title="Paired case differences; negative favors Fourier",
        ylabel="Absolute probability difference",
        xlabel="Independent case index",
    )
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.3f}"))
    axes[1].set_ylabel("Difference (percentage points)")
    axes[1].legend(fontsize=8)
    names = list(summary["groups"])
    x = np.arange(len(names))
    group_colors = colors | {"adaptive_spsa": "#aa7eb5", "adaptive_fourier": "#263a46"}
    for method, color in group_colors.items():
        y = [
            summary["groups"][g]["methods"].get(method, {}).get("mean_final_error", np.nan)
            for g in names
        ]
        if np.any(np.isfinite(y)):
            axes[2].plot(x, y, "o", color=color, label=LABELS[method].split(" ·")[0])
    axes[2].set_xticks(
        x,
        [g.replace("_", " ").replace(" / cloud verification", "\ncloud check") for g in names],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    axes[2].set(
        title="Every group: mean final error (extensions are exploratory)",
        ylabel="Failure probability (symlog; zeros retained)",
    )
    axes[2].set_yscale("symlog", linthresh=0.0001)
    axes[2].set_ylim(bottom=0)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("VFQEC · held-out calibration study", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _percent(value) -> str:
    return "unavailable" if value is None else f"{100 * value:.4f}%"


def _interval(values) -> str:
    return "unavailable" if values is None else f"[{100 * values[0]:.4f}, {100 * values[1]:.4f}] pp"


def _difference(value) -> str:
    return "unavailable" if value is None else f"{100 * value:.4f} pp"


def write_campaign_report(results: list[dict], out: Path) -> dict:
    """Save a complete summary, one figure, and self-contained HTML; return the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = analyze_campaign(results)
    payload = json.dumps(summary, indent=2, allow_nan=False)
    (out / "summary.json").write_text(payload)
    _figure(summary, out / "comparison.png")
    esc = html.escape
    comparisons = []
    for comparison in summary["primary"]["comparisons"].values():
        exact = comparison["secondary_exact_probability"]
        comparisons.append(
            "<article><h3>Fourier versus "
            + esc(LABELS[comparison["reference"]])
            + "</h3><p class='metric'>"
            + _difference(comparison["mean_difference"])
            + "</p><p>Mean paired difference; 95% interval "
            + _interval(comparison["ci95"])
            + ". Negative favors Fourier.</p><p>"
            + esc(comparison["conclusion"])
            + ".</p><p>Secondary, evaluation-only exact probability difference: "
            + _difference(exact["mean"])
            + "; 95% case-bootstrap interval "
            + _interval(exact["ci95"])
            + ". Exact probabilities do not select the controller.</p></article>"
        )
    table = []
    for group, values in summary["groups"].items():
        for method, metrics in values["methods"].items():
            table.append(
                "<tr>"
                + "".join(
                    "<td>" + esc(str(v)) + "</td>"
                    for v in (
                        group,
                        LABELS.get(method, method).split(" ·")[0],
                        metrics["cases"],
                        _percent(metrics["mean_final_error"]),
                        f"{metrics['total_final_errors']} / {metrics['total_evaluation_shots']}",
                        metrics["total_initial_calibration_shots"],
                        metrics["total_initial_probe_evaluations"],
                    )
                )
                + "</tr>"
            )
    cases = []
    for row in summary["case_results"]:
        methods = "; ".join(
            f"{LABELS.get(name, name).split(' ·')[0]}: {m['final_errors']}/{m['shots']} "
            f"({_percent(m['final_error'])})"
            for name, m in row["methods"].items()
        )
        cases.append(
            "<tr>"
            + "".join(
                "<td>" + esc(str(v)) + "</td>"
                for v in (
                    row["id"],
                    row["analysis_group"],
                    row["seed"],
                    row["backend"],
                    methods,
                )
            )
            + "</tr>"
        )
    calibration_rows = []
    for group, values in summary["groups"].items():
        for optimizer, cost in values["calibration"].items():
            calibration_rows.append(
                "<tr>"
                + "".join(
                    "<td>" + esc(str(v)) + "</td>"
                    for v in (
                        group,
                        LABELS.get(optimizer, optimizer).split(" ·")[0],
                        cost["cases"],
                        cost["initial_shots"],
                        cost["executed_shots"],
                        cost["executed_probe_evaluations"],
                        cost["calibrations"],
                        cost["unused_reference_recalibration_shots"],
                    )
                )
                + "</tr>"
            )
    ratio = summary["primary"]["initial_shot_ratios_reference_to_fourier"]["spsa_reference"]
    ratio_text = "unavailable" if ratio is None else f"{ratio:.3f}×"
    image = base64.b64encode((out / "comparison.png").read_bytes()).decode()
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='referrer' content='no-referrer'>"
        "<title>VFQEC — Held-out calibration study</title><style>"
        "body{margin:0;background:#f7f8f2;color:#173c37;font:16px/1.6 system-ui,sans-serif}"
        "main{max-width:1160px;margin:auto;padding:48px 24px}h1{font-size:42px;line-height:1.1}"
        "h2{margin-top:44px}.eyebrow{letter-spacing:.15em;font-size:12px;font-weight:700}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}"
        "article{background:white;padding:24px;border:1px solid #d9e2d7;border-radius:16px}"
        ".metric{font-size:32px;font-weight:700;margin:8px 0}.notice{border-left:4px solid "
        "#b57b46;padding:12px 20px;background:#fff5e4}img{max-width:100%;height:auto}"
        ".scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;"
        "font-size:13px}td,th{padding:12px;border-bottom:1px solid #dce3d9;text-align:left;"
        "vertical-align:top}th{white-space:nowrap}a{color:#087b61}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere;font-size:12px}li{margin-bottom:10px}</style></head><body><main>"
        "<p class='eyebrow'>VFQEC / RESEARCH EXTENSION</p>"
        "<h1>Can calibration use fewer probes?</h1>"
        "<p>Bounded Fourier coordinate calibration compared with budget-matched and reference "
        "SPSA. A model-specific exploratory study with a frozen held-out panel.</p>"
        "<p class='notice'>" + str(summary["primary"]["cases"]) + " of 24 primary independent "
        "cases present. "
        + (
            "The planned held-out seed panel is complete."
            if summary["primary"]["complete"]
            else "The panel is incomplete; no primary conclusion is reported."
        )
        + " Noninferiority margin: 0.001 absolute probability, or 0.1 percentage "
        "points. Neither noninferiority nor zero observed errors establishes equality.</p>"
        "<section class='cards'>"
        + "".join(comparisons)
        + "<article><h3>Initial calibration budget ratio</h3><p class='metric'>"
        + ratio_text
        + "</p><p>Total reference SPSA probe shots divided by Fourier probe shots, across "
        "the primary cases. This is a resource ratio, not a measured speedup.</p></article>"
        "</section><h2>All cases, all groups</h2><p>Cloud verification copies are kept separate "
        "and never increase the primary sample size. Group means weight cases equally; summed "
        "counts below are descriptive. The group panel uses a symmetric-log axis, linear below "
        "0.01% probability, so actual zeros stay at zero rather than receiving a fabricated "
        "positive floor. Other panels have linear axes.</p>"
        "<img src='data:image/png;base64," + image + "' alt='Per-seed error, paired differences, "
        "and group mean error comparison'><div class='scroll'><table><thead><tr><th>Group</th>"
        "<th>Method</th><th>Cases</th><th>Mean final error</th><th>Final counts</th>"
        "<th>Initial probe shots</th><th>Initial evaluations</th></tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table></div><h2>Every executed calibration</h2>"
        "<p>Initial-fit costs above and total executed costs below are distinct. Totals sum every "
        "recorded calibration update, including the initial update exactly once. The frozen drift "
        "runner recalibrated reference SPSA three times but evaluated only its static arm; later "
        "reference updates were unused. Their shots are retained, not erased from accounting. "
        "Static/adaptive arms share work, so costs are reported once per optimizer.</p>"
        "<div class='scroll'><table><thead><tr><th>Group</th><th>Optimizer</th><th>Cases</th>"
        "<th>Initial shots</th><th>All executed shots</th><th>All evaluations</th>"
        "<th>Calibrations</th><th>Unused reference update shots</th></tr></thead><tbody>"
        + "".join(calibration_rows)
        + "</tbody></table></div>"
        "<h2>Complete case outcomes</h2><div class='scroll'><table><thead><tr><th>Case ID</th>"
        "<th>Group</th><th>Seed</th><th>Backend</th><th>Every evaluated method</th></tr>"
        "</thead><tbody>" + "".join(cases) + "</tbody></table></div><h2>Interpretation limits</h2>"
        "<ul>" + "".join("<li>" + esc(n) + "</li>" for n in LIMITATIONS) + "</ul>"
        "<h2>Prior art and novelty boundary</h2><ul>"
        + "".join(
            "<li><a href='"
            + esc(p["url"], quote=True)
            + "'>"
            + esc(p["title"])
            + "</a>. "
            + esc(p["relevance"])
            + "</li>"
            for p in PRIOR_ART
        )
        + "</ul><details><summary>Complete machine-readable analysis</summary><pre>"
        + esc(payload)
        + "</pre></details></main></body></html>"
    )
    (out / "report.html").write_text(page)
    return summary
