## 快速开始（DRY_RUN）
1) 配置 `configs/reid.yaml` 的 `dataset.root`=[TBD]
2) 运行端到端：
```bash
bash scripts/run_all.sh --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --out outputs
```

### 产物

* captions.json（结构化属性/自然语混合）
* *_embeds.npy（文本/图像嵌入占位）
* metrics.json（包含 Rank-1 与 mAP）

### Testing
本地：
```bash
pytest -q
python tools/smoke_test.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --out outputs
```

### CI

已提供 `.github/workflows/ci.yml`。默认 `CI_MOCK=1`，自动生成最小 mock 子集并运行 smoke + pytest。将徽章指向仓库 Actions 运行结果。

### Ablation（小子集 2×2）

```bash
bash scripts/ablate.sh configs/reid.yaml "<DATASET_ROOT>" outputs/ablation
# Windows
scripts\\ablate.bat configs\
eid.yaml "<DATASET_ROOT>" outputs\\ablation
```

输出：`outputs/ablation/metrics.csv`，列：text_backend,image_backend,rank1,mAP,n_query,n_gallery,elapsed_s

### Observability
见 docs/Observability.md。
