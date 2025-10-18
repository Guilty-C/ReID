# Execution Summary — Phase 5–6
## 环境
引用：ARTIFACTS/env_report.txt（Python=<PY_VER>，GPU=<YES/NO>）
## 命令与耗时
- 命令：见 RUNBOOK.md
- 总耗时：[TBD]
## 指标
| rank1 | mAP | n_query | n_gallery |
|---:|---:|---:|---:|
| 0.25 | 0.0 | 8 | 8 |

## ISO 小节 — 基线登记
- 变体：`real`（文本/图像后端 `clip_l14`）
- 设备：`cuda`，批大小：`text=64`、`image=128`
- 指标：`rank1=0.25`、`mAP=0.0`、`n_query=8`、`n_gallery=8`
- 对照：`mock rank1=0.125`（提升），`iso_compare.csv` 已生成
- 摘要：见 `outputs/diagnostics/iso_summary_real.txt`

## CI & Tests
- CI：Ubuntu + Python 3.10/3.11，`CI_MOCK=1` 通过
- 本地测试：`pytest -q` 通过（见 ARTIFACTS/run_logs/<ts>/）

## Ablation 摘要
见 `outputs/ablation/metrics.csv`（4 组 2×2 组合）
## 新增文件树
V2/
  outputs/{captions.json,text_embeds.npy,img_embeds.npy,metrics/metrics.json}
  ARTIFACTS/run_logs/<ts>/run.log
  docs/{RUNBOOK.md,CHANGELOG.md,PR_DRAFT.md,Execution_Summary.md}
  deliverables_<YYYYMMDD-HHMM>.zip

### ISO Prompt 消融

| Prompt | Rank-1 | mAP | n_query | n_gallery | elapsed_s |
|---|---:|---:|---:|---:|---:|
| P1 | 0.1250 | 0.0000 | 8 | 8 | 134.291 |
| P2 | 0.1250 | 0.0000 | 8 | 8 | 129.863 |
| P3 | 0.1250 | 0.0000 | 8 | 8 | 131.618 |
| P4 | 0.1250 | 0.0000 | 8 | 8 | 130.102 |
| P5 | 0.1250 | 0.0000 | 8 | 8 | 128.250 |

Best (mAP): P1 (mAP=0.0000, rank1=0.1250)


### ISO Prompt 消融

| Prompt | Rank-1 | mAP | n_query | n_gallery | elapsed_s |
|---|---:|---:|---:|---:|---:|
| P1 | 0.1250 | 0.0000 | 8 | 8 | 66.236 |
| P2 | 0.0000 | 0.0000 | 8 | 8 | 66.536 |
| P3 | 0.1250 | 0.0000 | 8 | 8 | 66.114 |
| P4 | 0.1250 | 0.0000 | 8 | 8 | 68.554 |
| P5 | 0.0000 | 0.0000 | 8 | 8 | 67.924 |

Best (mAP): P1 (mAP=0.0000, rank1=0.1250)


### ISO Prompt 消融

| Prompt | Rank-1 | mAP | n_query | n_gallery | elapsed_s |
|---|---:|---:|---:|---:|---:|
| P1 | 0.0000 | 0.0000 | 8 | 8 | 48.113 |
| P2 | 0.0000 | 0.0000 | 8 | 8 | 32.032 |
| P3 | 0.1250 | 0.0000 | 8 | 8 | 30.552 |
| P4 | 0.0000 | 0.0000 | 8 | 8 | 29.086 |
| P5 | 0.0000 | 0.0000 | 8 | 8 | 27.698 |

Best (mAP): P3 (mAP=0.0000, rank1=0.1250)
Model= gpt-4o-mini, Language= English, Timestamp= 2025-10-18 19:34:26


### ISO Prompt 消融

| Prompt | Rank-1 | mAP | n_query | n_gallery | elapsed_s |
|---|---:|---:|---:|---:|---:|
| P1 | 0.1250 | 0.0000 | 8 | 8 | 20.179 |
| P2 | 0.0000 | 0.0000 | 8 | 8 | 20.829 |
| P3 | 0.1250 | 0.0000 | 8 | 8 | 18.859 |
| P4 | 0.0000 | 0.0000 | 8 | 8 | 17.768 |
| P5 | 0.0000 | 0.0000 | 8 | 8 | 17.127 |

Best (mAP): P1 (mAP=0.0000, rank1=0.1250)
Model= gpt-4o-mini, Language= English, Timestamp= 2025-10-18 19:39:44


### Evaluator Correction (ID-aware)
- Uses full Nq×Ng similarity and ID arrays for Rank-1/mAP.
- Metrics (schema): Rank-1=0.1250, mAP=0.3211, n_query=8, n_gallery=8
- Captions enforce strict English JSON schema; sanitize to a small vocabulary.
- Artifacts: captions JSON(raw/sanitized), phrases, aligned IDs, metrics JSON.


### Micro-cell Prompt Evaluation
Best prompt: P1 (avg mAP=0.3397, avg Rank-1=0.1250)

| Prompt | avg_mAP | std_mAP | avg_Rank-1 | std_Rank-1 | cells |
|---|---:|---:|---:|---:|---:|
| P1 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P2 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P3 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P4 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P5 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |

See ablation/microcell_metrics.csv and ablation/microcell_summary.csv for details.


### Micro-cell Prompt Evaluation
Best prompt: P1 (avg mAP=0.3397, avg Rank-1=0.1250)

| Prompt | avg_mAP | std_mAP | avg_Rank-1 | std_Rank-1 | cells |
|---|---:|---:|---:|---:|---:|
| P1 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P2 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P3 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P4 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |
| P5 | 0.3397 | 0.0000 | 0.1250 | 0.0000 | 1 |

See ablation/microcell_metrics.csv and ablation/microcell_summary.csv for details.


### Larger-ISO Evaluation (Best Prompt)
| Size | Rank-1 | mAP | n_query | n_gallery | elapsed_s |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.1875 | 0.3138 | 32 | 32 | 0.01 |
| 64 | 0.0781 | 0.2186 | 64 | 64 | 0.13 |

Recommendation: iterate (vs micro-cell avg mAP=0.3397, Rank-1=0.1250).
