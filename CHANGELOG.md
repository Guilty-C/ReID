# 变更日志

## [1.0.0] - 2025-10-17
### 新增
- 完整的PRP-ReID项目结构
- 核心工具脚本：captioner.py, embed_text.py, embed_image.py, retrieve_eval.py
- 配置文件：reid.yaml
- 运行脚本：run_all.sh, run_all.bat
- 测试框架：pytest测试用例
- CI/CD配置：GitHub Actions
- 消融实验脚本
- 文档：实现计划、WBS、假设说明等

### 功能
- 支持三种描述模式：自然语言、显著特征、结构化JSON
- 基于CLIP的文本和图像嵌入
- Rank-1和mAP评估指标
- 2×2消融实验框架
- 端到端测试验证