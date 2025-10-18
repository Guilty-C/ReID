# API 接入配置（Aizex）

## 基础路由（BASE_URL）
- 主站（优先）：`https://aizex.top/v1/chat/completions`
- 国内直连优化（仅 /v1 白名单）：`https://a1.aizex.me/v1/chat/completions`
- 若使用 OpenAI SDK 的 `base_url` 方式，请按提供商指引分别尝试：根域名、`/v1`、`/v1/chat/completions`。

## 环境变量
在 PowerShell（Windows）中设置：
```powershell
$env:CAPTION_API_URL = 'https://aizex.top/v1/chat/completions'
$env:CAPTION_API_KEY = '<YOUR_AIZEX_API_KEY>'
$env:CAPTION_API_MODEL = 'gpt-4o-mini'  # 文本+图像，已验证
```
说明：
- 代码自动使用 `Authorization: Bearer <KEY>` 进行鉴权。
- 若 KEY 前含有 `api:` 前缀，代码会自动剥离；请按原样粘贴即可。

## 模型调用名（重要）
- 请在 Aizex 控制台左侧「模型价格」中确认实际调用名。
- 若出现 `model_not_found` / “分组 default 下模型 XXX 无可用渠道（distributor）”，说明当前分组未开通该模型，请更换为可用模型或调整分组配置。
- 本仓库已在 Aizex 验证：`gpt-4o-mini` 可用，并能稳定返回中文短语；`qwen-plus / qwen-vl-plus / qwen2.5-vl-plus` 在默认分组未找到渠道。

## 健康检查
```powershell
python tools\raw_http_test.py      # 纯文本消息连通性与鉴权
python tools\api_test.py           # 图像+文本字幕生成端到端
```
预期：返回 200 且 `content` 为中文属性短语。

## 常见错误与处理
- 401：API Key 不正确或无效。请重新粘贴或在控制台重置。
- 404：路径未找到。请尝试切换到 `.../v1/chat/completions`。
- 503 `model_not_found`：更换为控制台可用的模型调用名，或在 Aizex 后台为该分组开通渠道。

## 评测运行
环境就绪后，直接运行：
```powershell
python tools\iso_prompt_ablate.py
```
产出：
- 字幕：`outputs/captions/iso_api_P*.json`
- 指标：`outputs/ablation/iso_prompt_metrics.csv`
- 质量概览：`outputs/ablation/caption_quality.csv`
- 文档：`docs/Execution_Summary.md`（自动追加消融表格）