# PRP-ReID: 基于属性描述的文本到图像行人重识别系统

## 项目概述
本项目实现了一个端到端的文本到图像行人重识别（ReID）系统，支持从自然语言描述到图像检索的完整流程。

## 核心功能
- **图像描述生成**: 支持自然语言、显著特征、结构化JSON三种描述模式
- **文本/图像嵌入**: 基于CLIP模型的向量化表示
- **检索评估**: 支持Rank-1和mAP评估指标
- **消融实验**: 2×2组合测试不同CLIP后端

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 端到端运行
bash scripts/run_all.sh

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

详细使用说明请参考 `docs/` 目录下的文档。