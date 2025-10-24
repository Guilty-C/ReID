# Baseline Freeze (2025-10-24)

- Baseline: `outputs/comparison/20251024-190356_attrbank_v3_lr/`
- Model: CLIP ViT-L/14, center crop, L2
- Filelists: `larger_iso/64/query.txt`, `larger_iso/64/gallery.txt`
- z-score: global mean/variance
- Fusion: Logistic Regression (5-fold, class_weight=balanced) + Platt
- Metrics: Rank-1=0.06, mAP=0.1302 (+13.3%), nDCG@10=0.1456 (+18.4%)
- Re-ranking: disabled (no gain this round)
- Seed: 42

Decision: Treat `attrbank_v3_lr` as current baseline; keep re-ranking off until recall improves.