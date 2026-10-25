# F1 study protocol — October 25, 2026

Frozen before running the held-out panel. The canonical settings and every seed
are in `protocol.json`. Primary inference uses all 24 cases with seeds 2001–2024;
development and exploratory extensions are excluded from that inference.

## Question

Can bounded, validation-gated Fourier coordinate calibration reduce calibration
probe shots while preserving final logical-memory performance in this model?
This is a software implementation and benchmark of a Rotosolve-derived method,
not a claim to have invented sinusoidal optimization or syndrome-based control.

Each coordinate is probed at -0.7, 0, +0.7 radians, followed by fresh baseline and
candidate validation probes. The candidate is accepted only if its independent
observed syndrome cost is lower. This rule is empirical, not a confidence
certificate. All three optimizers see only finite-shot syndrome observations.
Every probe, including validation, counts toward the reported learning budget.

All learning uses one-round probes: each control occurs once, permitting the
first-harmonic coordinate model. Historical E1 used a two-round objective and is
retained separately; this campaign is not a rerun of its original settings.

## Comparators and primary endpoint

- Fourier: four coordinate passes, 1,000 shots per observation, 60,000 total
  learning shots in three dimensions, within a 61,000-shot cap.
- Budget SPSA: 20 iterations, 1,000 shots per observation, 61,000 learning shots.
- Reference SPSA: 50 iterations, 4,000 shots per observation, 604,000 learning
  shots, with the original optimizer gains and averaging unchanged.
- Standard QEC, twirling, unencoded memory, and the single-qubit-field oracle
  remain explicit physical comparators. Drift adds adaptive SPSA and Fourier.

Primary endpoint: final observed logical-error probability, Fourier minus
reference SPSA, paired by the same physical configuration. Use a seed-level
10,000-replicate bootstrap with fixed RNG seed and a two-sided 95% interval.
The predeclared absolute noninferiority margin is 0.001 (0.1 percentage point).
Only an upper bound strictly below this margin supports model-specific
noninferiority in this panel. Also report the equal-budget SPSA comparison.
Exact density probabilities are evaluation-only secondary diagnostics, never
inputs to candidate selection. Do not bootstrap rounds as independent trials.

Exploratory panels cover X memory, measurement noise, distance 5/7, drift, and
both d=3 surface memory bases with/without ZZ crosstalk. Surface cases use lower
shots and shorter memory and do not include the expensive reference SPSA arm.
No adjustment of settings after held-out outcomes and no outcome-based omission.

## Cloud cross-check and money

Predeclared CPU cloud cross-checks: heldout_static-2001 and phase_memory-3001.
These are backend consistency checks, not extra independent primary seeds.
BlueQubit supplies coherent simulation layers; noise, syndrome extraction,
recovery, and observation sampling remain local. This is not QPU execution.

The new campaign authorization is at most USD 10 total. The SDK has no hard
per-job price-cap option. A persistent shared guard therefore admits only
freshly quoted USD 0 CPU jobs, disables hidden SDK submission retries, records
every admitted job, and freezes further work on ambiguous submission or
unknown/nonzero actual cost. At most 800 cloud submissions are allowed.
No paid GPU or QPU work is submitted. See the provider's
[device documentation](https://app.bluequbit.io/docs) and
[pricing](https://www.bluequbit.io/pricing).

## Prior work and limits

[Rotosolve](https://arxiv.org/abs/1905.09692),
[syndrome noise estimation](https://arxiv.org/abs/2010.02243),
[in-situ coherent calibration](https://arxiv.org/abs/2511.01080), and
[fast-feedback drift control](https://arxiv.org/abs/2512.07815) precede this study.
An improvement here would be an empirical software finding under the stated
model. It would not establish global novelty, a fault-tolerant threshold,
hardware gains, or quantum advantage. Calibration freezes memory time; gates
and extraction are ideal, and the surface decoder is spatial, not spacetime.
