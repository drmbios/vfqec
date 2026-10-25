# F1: fewer calibration probes, with a measured accuracy trade-off

Study date: October 25, 2026. The [frozen protocol](PROTOCOL.md) was committed
before examining the held-out panel, at `d73951d`. Every planned case is retained.

## Main result

| Controller | Initial learning shots per case | Final failures / 96,000 readouts |
|---|---:|---:|
| Validation-gated Fourier | 60,000 | 75 |
| Budget SPSA | 61,000 | 72 |
| Reference SPSA | 604,000 | 61 |
| Standard QEC | — | 3,119 |
| Twirled QEC | — | 2,228 |
| Single-qubit-field oracle | — | 62 |

The Fourier controller uses **10.067× fewer initial learning shots** than the
reference. Its mean paired final error difference is **+0.000145833** (Fourier
minus reference), with 95% seed-level bootstrap interval
**[−0.000114583, +0.000395833]**. This satisfies the prospectively chosen
**0.001 absolute** noninferiority margin in these 24 configurations. It does
not establish superiority or identical accuracy. The margin is 0.1 percentage
point, not a 0.1% relative difference.

Evaluation-only exact density probabilities favor reference SPSA slightly:
mean difference **+0.000091052**, interval **[+0.000064707, +0.000120116]**.
The budget-SPSA observed comparison is inconclusive: mean **+0.00003125**,
interval **[−0.000208333, +0.00028125]**. There is no demonstrated equal-budget
accuracy advantage. Confidence intervals resample independent cases, never
correlated round endpoints.

## What the software adds

`FourierOptimizer` fits a first-harmonic coordinate slice from three finite-shot
syndrome probes at −0.7, 0, +0.7 radians. It chooses a bounded minimum and uses
two fresh, independent observations to decide whether to accept the proposal.
Every fitting and validation shot is counted. The default acceptance rule is
empirical; it is not a statistical confidence certificate. A conservative optional
Hoeffding gate uses independent probe shots rather than correlated check bits.

All campaign optimizers use **one-round probes**, where each control angle
appears once. Historical E1 used a two-round objective. This study therefore
does not claim the same budget comparison against that original objective.
Logical outcomes, exact probabilities, and true fields are excluded from the
optimizer interface and added only after calibration for evaluation.

These are **simulated** finite-shot observations. BlueQubit supplies coherent
statevector layers; noise, syndrome measurement, recovery and finite-shot
sampling run locally. The 10.067× figure does not establish hardware shot,
cloud-credit, or wall-clock savings.

## Complete coverage and negative results

The 48-case panel includes four development cases; 24 held-out configurations;
X memory; noisy syndrome reporting; distances 5 and 7; drift; and both surface
memory bases with and without ZZ crosstalk. Extension groups are small and
exploratory. Adaptive control does not consistently improve drift outcomes.
The surface X-memory case with ZZ remains poor even for the single-qubit oracle.
Single-qubit compensation cannot remove the ZZ term, which is outside the available control set.

Drift execution includes repeated reference-SPSA calibrations even though only
its initial field is evaluated. Those **4,832,000 unused reference learning
shots** are retained in the full accounting, rather than excluded after seeing
results. Initial and complete executed costs are reported separately.

All 18 original full E1–E4 runs also completed, using 24,861,064 conceptual
shots across separately capped suites. E1 reproduces the original six counts
1029, 22, 16, 2, 2, 2 out of 4,000. E2-X shows little improvement; E3 brackets
no crossing and establishes no threshold. In E4, slow updates can be worse
than calibrating once. E5 remains an unsubmitted hardware template.

## Cloud cross-check, cost and provenance

Both predeclared cases completed: **511 CPU jobs**, all settled at **USD 0.00**
against the **USD 10 maximum** authorization. All learned control vectors match
the local reproductions exactly. Per-method probability and sampled-readout
differences are preserved in the verification record.

The two predeclared verification cases are `heldout_static-2001` and
`phase_memory-3001`. They repeat local physical cases and do not increase the
primary sample size. [Cloud verification](cloud-verification.json) records each
comparison, all admitted job IDs and settled actual costs. A shared SQLite
guard admits only fresh $0 CPU estimates, caps submissions at 800 and total
authorization at $10, disables SDK submission retries, and stops on ambiguous
submission or unknown/nonzero actual cost. No paid GPU or QPU work is submitted.

Raw `case.config` describes the planned physical case and retains `backend=local`;
the result's backend label and verification record identify actual cloud execution.
The raw execution commit may contain `+dirty`: reports and the website were
being authored concurrently. The frozen numerical optimizer and physics sources
were unchanged during the study. The protocol commit identifies that implementation;
later packaging, dependency-security, reporting and UI changes are separate.

## Evidence and reproduction

- [Complete report](report.html) and [comparison figure](comparison.png)
- [Summary with every case](summary.json)
- [All 48 raw local cases](local-results.json.gz)
- [Raw cloud reproductions](cloud-results.json.gz)
- [All 18 raw legacy runs](legacy-results.json.gz) and [suite summary](legacy-summary.json)
- [Frozen machine-readable protocol](protocol.json)
- [SHA-256 artifact manifest](manifest.json)

Archives are gzip-compressed JSON arrays. Run the commands in the repository
README to reproduce the panels. `scripts/publish_research.py` refuses incomplete
local/cloud evidence; `scripts/build_demo.py` verifies artifact checksums and
deploys only an explicit file allowlist. Credentials are never published.

## Prior art and limits

[Ostaszewski et al.](https://arxiv.org/abs/1905.09692) introduced the Rotosolve
coordinate principle used here. [Wagner et al.](https://arxiv.org/abs/2010.02243),
[Kunjummen and Taylor](https://arxiv.org/abs/2511.01080), and
[Magann et al.](https://arxiv.org/abs/2512.07815) establish relevant syndrome-based
estimation and in-situ calibration precedents. This is an empirical software
extension, not a substantiated first-of-kind invention.

The model has ideal extraction/recovery gates, frozen calibration time, and
spatial decoding. One memory basis is not full quantum process tomography.
No hardware performance, quantum advantage, or fault-tolerant threshold follows.
