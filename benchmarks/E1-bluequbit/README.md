# E1: BlueQubit CPU hybrid benchmark

Executed on October 1, 2026. Run ID: `aca95affb4e946598029f8c307c01b80`.
Source commit: `f2f026c2021c444a2a36edd1972eb47c77148676`. Seed: 1729.

BlueQubit executed the coherent Choi probes. Stochastic noise, finite-shot syndrome
measurement, decoding, and recovery ran locally. This benchmark is simulation;
no QEC hardware execution is claimed. The numerical optimizer uses syndrome
statistics only, with the rule-based agent orchestration fallback.

## Configuration and result

Three-qubit repetition memory; hidden X rotations `(0.12, -0.08, 0.15)` radians;
stochastic X error probability `p=0.002`; ideal syndrome measurement `q=0`;
60 rounds; 50 SPSA steps per calibration; 4,000 shots per cost evaluation and
4,000 endpoint readouts per method; adaptive cadence 20 rounds.

| Method | Final logical failures / 4,000 | Observed error probability |
|---|---:|---:|
| Unencoded qubit | 1,029 | 25.725% |
| Standard QEC | 22 | 0.55% |
| QEC + twirling | 16 | 0.40% |
| QEC + calibrated field | 2 | 0.05% |
| QEC + adaptive field | 2 | 0.05% |
| Oracle | 2 | 0.05% |

Initial learned corrections: `(0.12834958, -0.08867223, 0.14366648)` radians.
The learned method has 11 times fewer observed final failures than standard QEC
in this run and matches the oracle's observed final count. These are finite-shot
observations from one seed/model, not a hardware-performance or universal-improvement claim.

All 458 remote jobs completed. The SDK recorded total cost of $0.00.
The experiment accounted for 3,252,000 shots, within a $10 estimated cloud-cost
cap and 1,000-job cap. Prices and provider availability may change.

## Reproduction

From the repository root, install dependencies as described in the main README,
configure `BLUEQUBIT_API_TOKEN` through `.env` or your environment, then run:

```bash
vfqec run E1 --backend bluequbit --device cpu --max-credits 10 --max-remote-jobs 1000
vfqec run E1 --backend local
```

The 453 optimizer observations and all learned field updates match the
[included local reference](../E1-local/result.json) exactly. All six final logical
counts also match. Intermediate numerical differences include at most one
unencoded logical failure and eight oracle syndrome events; the largest exact
probability difference is about 1.33e-6.

## Evidence

- [Full result, configuration, optimizer history, and provider job IDs](result.json)
- [Comparison and recorded-cost verification](cloud-verification.json)
- [PDF report](report.pdf) and [standalone HTML report](report.html)
- [Artifact checksums](checksums.json)
- [Exported repetition syndrome circuit](syndrome-circuit.qasm)

The exported circuit is the syndrome-extraction circuit; the remote jobs used
coherent Choi probes as described above. Internal cache paths in the result JSON
refer to the execution container, not files shipped in this benchmark.

![Six-method logical error comparison](logical-error.png)
![Syndrome-only optimizer convergence](convergence.png)
![Learned corrections and evaluation-only true fields](fields.png)
