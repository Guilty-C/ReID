# Requirements Map

RQ-01 — Environment Baseline
- Evidence: docs/ENV_REPORT.json

RQ-02 — Acceptance (Metrics Gate)
- Policy: Rank-1 ≥ 50%, mAP ≥ 50%
- Metrics source-of-truth: outputs/metrics/metrics.json
- Gate record: submission/metrics_gate.json

RQ-03 — Operational Evidence
- E2E log: runs/logs/E2E_2025-10-19T15-00+09-00.md
- Outputs manifest: outputs/MANIFEST_2025-10-19.txt

Update process:
- If multiple candidates exist, selection rule: prefer canonical path produced by scripts/run_all.sh (outputs/metrics/metrics.json).
- If no metrics appear, add an OPEN_QUESTION and proceed to Step 4 with available outputs.