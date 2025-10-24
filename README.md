# PRP-ReID: 基于属性描述的文本到图像行人重识别系统

## 项目概述
本项目实现了一个端到端的文本到图像行人重识别（ReID）系统，支持从自然语言描述到图像检索的完整流程。

## 核心功能
- **图像描述生成**: 支持自然语言、显著特征、结构化JSON三种描述模式，新增支持通过外部API生成（LLM 视觉-文字）。
- **文本/图像嵌入**: 基于CLIP模型的向量化表示。
- **检索评估**: 支持Rank-1和mAP评估指标；支持ID-aware评估（基于 query/gallery 的ID文件）。
- **消融实验**: 2×2组合测试不同CLIP后端。

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 端到端运行（API 图像描述→向量化→检索评估）
python tools/run_experiment.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" \
  --prompt-file prompts/person_desc.txt --api-url "<API_URL>" --api-key "<API_KEY>" --api-model "<MODEL>" \
  --subset gold --device cpu

# 运行测试
pytest tests/
```

## 项目结构
```
├── configs/          # 配置文件
├── tools/            # 核心工具脚本
├── scripts/          # 运行脚本
├── tests/            # 测试框架
├── docs/             # 文档
└── outputs/          # 输出目录
```

## 进度（Progress）
- 环境基线：见 `docs/PROCESS.md` 与 `docs/ENV_REPORT.json`。
- LLM Caption Experiment：
  - 运行：`python tools/run_experiment.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --prompt-file prompts/person_desc.json --api-url "<API_URL>" --api-key "<API_KEY>" --api-model "<MODEL>" --subset gold`。
  - 产物：`outputs/experiments/EXP_*/{captions/captions_query.json,embeds/text_q.npy,embeds/img_g.npy,embeds/query_ids.npy,embeds/gallery_ids.npy,embeds/similarity_qg.npy,metrics/metrics.json,run.md,run_summary.json}`。
  - ID-aware：评估消费 `embeds/query_ids.npy` 与 `embeds/gallery_ids.npy`。
  - 指标：Rank-1 / mAP 写入 `metrics/metrics.json`；每步 stdout 写入 `outputs/experiments/EXP_*/logs/*.txt`。
  - API 配置：可通过 CLI 传入或使用环境变量 `CAPTION_API_URL`、`CAPTION_API_KEY`、`CAPTION_API_MODEL`。
  - Prompt 文件：使用 `prompts/person_desc.json`（支持 JSON/文本；优先读取 JSON 的 `prompt`/`text`/`instruction` 键；路径记录在 `captions_query_meta.json`）。
  - 子集与路径：使用 `--subset gold` 固定小样本；gallery 路径写入 `embeds/gallery_paths.txt`，并基于这些路径生成 `gallery_ids.npy`，确保 ID 与子集对齐。
  - 环境修复：运行器在调用 captioner 前注入 `CAPTION_API_URL`、`CAPTION_API_KEY`、`CAPTION_API_MODEL` 环境变量，修复此前 raw 日志中的 `[API_ERROR] missing env`。

详细使用说明请参考 `docs/` 目录下的文档。