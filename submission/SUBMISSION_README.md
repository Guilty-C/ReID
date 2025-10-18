# Submission README — 2025-10-19

Context
- Date: 2025-10-19
- Seed: 0
- Commit: unverified
- Branch: unverified
- Environment summary (docs/ENV_REPORT.json): Python 3.11.9; Torch 2.5.1+cu121; CUDA 12.1; platform/GPU unverified

Metrics
- Source-of-truth: outputs/metrics/metrics.json
- Raw values: rank1=0.00025342118601115053, mAP=0.0 (fractions)
- Normalization: fractions × 100 → percent
- Rank-1 (percent): 0.025342118601115053
- mAP (percent): 0.0
- Gate thresholds: Rank-1 ≥ 50%, mAP ≥ 50%
- Gate outcome: FAIL

Contents (submission/)
- metrics_gate.json — gate record (timestamp, thresholds, metrics, verdict)
- SUBMISSION_README.md — this file
- MANIFEST_2025-10-19.txt — outputs manifest
- ENV_REPORT.json — copied from docs/ENV_REPORT.json
- E2E_2025-10-19T15-00+09-00.md — E2E run log
- SMOKE_2025-10-19T10-00+09-00.md — smoke run log
- metrics.json — canonical metrics file (copy of outputs/metrics/metrics.json)

Reproduce notes (offline-only)
- Canonical entry: scripts/run_all.sh
- Config: configs/reid.yaml
- DATA_ROOT: D:/PRP SunnyLab/ReID/data/archive/Market-1501-v15.09.15
- Run within the 15:00–16:30 JST window; ensure outputs/metrics/metrics.json is produced; document any deviations.