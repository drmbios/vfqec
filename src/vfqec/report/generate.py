"""Generate the same six-method comparison in PNG, Markdown, HTML and PDF."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from vfqec.report.analysis import wilson

LABELS = {
    "unencoded": "Unencoded qubit",
    "standard": "Standard QEC",
    "twirled": "QEC + twirling",
    "calibrated": "QEC + calibrated field",
    "adaptive": "QEC + adaptive field",
    "oracle": "Oracle (single-qubit field)",
}


def figures(result: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(1, result["config"]["rounds"] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    upper = 0.0
    for method, values in result["methods"].items():
        lo, hi = wilson(values["errors"], values["shots"])
        for ax in axes:
            (line,) = ax.plot(rounds, values["logical_error"], label=LABELS[method], linewidth=1.5)
            ax.fill_between(rounds, lo, hi, color=line.get_color(), alpha=0.08)
        if method != "unencoded":
            upper = max(upper, float(np.max(hi)))
    axes[0].set(title=result["backend"], ylabel="Logical failure probability", ylim=(0, 1))
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set(
        title="Same six curves: zoom to encoded error scale",
        ylim=(0, max(0.002, upper)),
        xlabel="QEC round (hypothetical terminal readout)",
        ylabel="Logical failure probability",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "logical-error.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for start in sorted({r["calibration_round"] for r in result["optimizer"]}):
        rows = [
            r
            for r in result["optimizer"]
            if r["calibration_round"] == start and r["phase"] in ("initial", "candidate")
        ]
        ax.plot(
            [r["shots_used"] for r in rows],
            [r["cost"] for r in rows],
            label=f"Calibration at round {start}",
        )
    ax.set(
        xlabel="Calibration shots",
        ylabel="Measured syndrome firing rate",
        title=f"Syndrome-only optimizer | {result['backend']}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=160)
    plt.close(fig)
    true = np.array(result["true_fields"])
    updates = result["updates"]
    adaptive = np.array(
        [updates[t // result["config"]["cadence"]]["theta"] for t in range(len(rounds))]
    )
    fig, axes = plt.subplots(2 if true.shape[1] > 9 else 1, 1, figsize=(10, 6), squeeze=False)
    for i in range(true.shape[1]):
        ax = axes[i // 9, 0]
        (line,) = ax.plot(rounds, true[:, i], linestyle="--", alpha=0.8, label=f"true {i}")
        ax.step(rounds, adaptive[:, i], color=line.get_color(), label=f"adaptive {i}", where="post")
        ax.axhline(updates[0]["theta"][i], color=line.get_color(), linestyle=":", alpha=0.6)
        ax.set(xlabel="Round", ylabel="Angle (rad)")
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle(
        f"Evaluation-only ground truth; dotted = calibrated | {result['backend']}", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "fields.png", dpi=160)
    plt.close(fig)
    return ["logical-error.png", "convergence.png", "fields.png"]


def generate_report(result: dict, out: Path) -> Path:
    images = figures(result, out)
    title = f"VFQEC {result['config']['experiment']} - {result['run_id']}"
    rows = [["Method", "Final error", "95% interval", "Fitted / round"]]
    for method, fit in result["analysis"]["fits"].items():
        lo, hi = fit["final_ci95"]
        rows.append(
            [
                LABELS[method],
                f"{fit['final_error']:.6g}",
                f"[{lo:.4g}, {hi:.4g}]",
                f"{fit['rate']:.6g}" if fit["rate"] is not None else "insufficient data",
            ]
        )
    notes = result["notes"] + [result["analysis"]["fit_caution"]] + result["analysis"]["anomalies"]
    markdown = [
        f"# {title}",
        "",
        f"Backend: **{result['backend']}**",
        "",
        f"Seed: {result['config']['seed']}; source: `{result['commit']}`; "
        f"shots charged to budget: {result['shots_used']}",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "|---|---|---|---|",
    ]
    markdown += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    markdown += ["", *[f"- {note}" for note in notes], ""]
    markdown += [f"![{image}]({image})" for image in images]
    markdown += ["", "## Configuration", "```json", json.dumps(result["config"], indent=2), "```"]
    (out / "report.md").write_text("\n".join(markdown))
    table_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}"
        "td{padding:10px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>"
        + html.escape(title)
        + "</h1><p>"
        + html.escape(result["backend"])
        + "</p><table>"
        + table_html
        + "</table><ul>"
        + "".join("<li>" + html.escape(n) + "</li>" for n in notes)
        + "</ul>"
        + "".join(
            "<img src='data:image/png;base64,"
            + base64.b64encode((out / name).read_bytes()).decode()
            + "' "
            f"alt='{name}'>"
            for name in images
        )
        + "</html>"
    )
    (out / "report.html").write_text(page)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(html.escape(title), styles["Title"]),
        Paragraph(html.escape(result["backend"]), styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]
    table = Table(rows, colWidths=[2.0 * inch, 0.8 * inch, 1.4 * inch, 1.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    for note in notes:
        story.extend([Spacer(1, 0.08 * inch), Paragraph(html.escape(note), styles["Normal"])])
    for name in images:
        story.extend(
            [
                Spacer(1, 0.15 * inch),
                Image(
                    str(out / name),
                    width=6.2 * inch,
                    height={"logical-error.png": 4.96, "convergence.png": 2.48, "fields.png": 3.72}[
                        name
                    ]
                    * inch,
                ),
            ]
        )
    SimpleDocTemplate(str(out / "report.pdf"), title=title).build(story)
    return out / "report.html"


def sweep_figure(results: list[dict], out: Path) -> Path | None:
    """Plot all six methods for distance/noise sweeps or calibration-budget sweeps."""
    if not results:
        return None
    distances = sorted({r["config"]["distance"] for r in results})
    noise_sweep = len({r["config"]["p"] for r in results}) > 1 or len(distances) > 1
    fig, axes = plt.subplots(
        1,
        len(distances) if noise_sweep else 1,
        figsize=(6 * len(distances) if noise_sweep else 9, 5),
        squeeze=False,
    )
    methods = list(results[0]["methods"])
    for index, distance in enumerate(distances if noise_sweep else [None]):
        ax = axes[0, index]
        selected = [r for r in results if distance is None or r["config"]["distance"] == distance]
        for method in methods:
            if noise_sweep:
                pairs = sorted(
                    (r["config"]["p"], r["methods"][method]["logical_error"][-1]) for r in selected
                )
            else:
                pairs = sorted(
                    (
                        sum(u["shots"] for u in r["updates"]),
                        r["methods"][method]["logical_error"][-1],
                    )
                    for r in selected
                )
            x, y = zip(*pairs)
            ax.plot(x, y, marker="o", label=LABELS[method])
        ax.set(
            xlabel="Physical Pauli probability p" if noise_sweep else "Total calibration shots",
            ylabel="Final logical failure probability",
            title=f"Distance {distance}"
            if noise_sweep
            else "Cadence and calibration-budget trade-off",
            ylim=(0, None),
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle(" | ".join(sorted({r["backend"] for r in results})), fontsize=9)
    fig.tight_layout()
    path = out / "sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
