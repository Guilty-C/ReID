## Environment Baseline — 2025-10-19T09:30+09:00
OS: Windows 10 (Git Bash)
Python: 3.11.9
Torch: 2.5.1+cu121
CUDA available: TBD
GPU: RTX 4060
DATA_ROOT: D:/PRP SunnyLab/ReID/data/archive/Market-1501-v15.09.15
Lockfile: docs/requirements.lock.txt
Env report: docs/ENV_REPORT.json

## Operational Notes
- 2025-10-19T10:00+09:00 — Smoke run scheduled; log scaffold at runs/logs/SMOKE_2025-10-19T10-00+09-00.md
- 2025-10-19T15:00+09:00 — E2E run scheduled; log scaffold at runs/logs/E2E_2025-10-19T15-00+09-00.md; canonical entry scripts/run_all.sh
- 2025-10-19T16:30+09:00 — Gate FAIL; elapsed 01:30:00; metrics at outputs/metrics/metrics.json; gate record at submission/metrics_gate.json

## LLM Caption Experiment
- Run: `python tools/run_experiment.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --prompt-file prompts/person_desc.txt --api-url "<API_URL>" --api-key "<API_KEY>" --api-model "<MODEL>" --subset gold`
- Artifacts: `outputs/experiments/EXP_*/{captions/captions_query.json,embeds/text_q.npy,embeds/img_g.npy,embeds/query_ids.npy,embeds/gallery_ids.npy,embeds/similarity_qg.npy,metrics/metrics.json,run.md,run_summary.json}`
- ID-aware: evaluation consumes `embeds/query_ids.npy` and `embeds/gallery_ids.npy`
- Rank-1 and mAP: recorded in `metrics/metrics.json`
- Logs: prompt and raw API responses saved next to `captions_query.json`