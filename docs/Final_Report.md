# 最终报告 (Final Report)

## 指标说明
**全量数据评估结果**：基于19,732张画廊图像和3,368个查询文本的完整数据集评估。


## 1. 执行摘要
- Rank-1: 0.0
- mAP: 0.0
- n_query: 3,368
- n_gallery: 19,732
- run_id: full_dataset_20251018

## 2. 方法与流水线
见 RUNBOOK.md 与实现概述（不改范围）。

## 3. 基准测试
**来源**：outputs/bench/metrics.csv

### 指标表
| text_backend | image_backend | rank1 | mAP | n_query | n_gallery |
|---|---|---|---|---|---|
| clip | b16_clip_b16 | 0.0 | 0.0 | 3368 | 19732 |
| clip | b16_clip_l14 | 0.0 | 0.0 | 3368 | 19732 |
| clip | l14_clip_b16 | 0.0 | 0.0 | 3368 | 19732 |
| clip | l14_clip_l14 | 0.0 | 0.0 | 3368 | 19732 |


## 4. 误差分析
- Top-K 列表：outputs/error_slices/topk_indices.txt
- 失败样例：outputs/error_slices/topk.csv
- 说明：若 id 对齐未完成，详细误差解读保留 [TBD]。

## 5. 校准与重排序
- 温度扫描：outputs/calib/metrics.csv
- 重排序结论：
[TBD: Re-ranking 结论缺失]

## 6. 稳定性
- 参见 docs/Benchmark_Stability.md（若缺则 [TBD]），口径为固定 SEED 多次复测的 mean/std。

## 7. 合规性
# Compliance Documentation

## Software Bill of Materials (SBOM)
- File: sbom.json
- Generated: 2025-10-18 01:25:19

## Third-Party Dependencies
- File: THIRD_PARTY.md
- Total packages: 84

## License Information
- [TBD] Add license compliance details

- SBOM：[TBD]
- 第三方依赖：THIRD_PARTY.md

## 8. 可复现性
- 关键命令：RUNBOOK.md
- 质量门禁：tools/quality_gate.py（CI 已集成）
- 运行历史（最近10行）:
[TBD: outputs/metrics/history.csv 缺失]

## 9. 发布说明
# Release Notes

## [Unreleased] Phase 5–6 — 最小复现与交付整合
- 新增：outputs/{captions.json,text_embeds.npy,img_embeds.npy,metrics.json}
- 新增：ARTIFACTS/run_logs/<ts>/run.log
- 新增：docs/{RUNBOOK.md,PR_DRAFT.md}
- 新增：deliverables_<YYYYMMDD-HHMM>/{文件集} 与同名 .zip

## 10. 附录
- data_contract.schema.json（如有）
- configs/reid.yaml / configs/reid_best.yaml（如有）
