# Decisions — 2025-10-19

- 2025-10-19T16:45+09:00 — Gate FAIL (Rank-1=0.0253%, mAP=0.0%). Decision: schedule a re-run window.

# Decisions — 2025-10-21

- 2025-10-21T11:30+09:00 — 切换字幕到远端 Aizex API；配置 `CAPTION_API_URL`、`CAPTION_API_KEY`、`CAPTION_API_MODEL`。
- 2025-10-21T11:35+09:00 — 主站健康检查返回 200（`https://aizex.top/v1/chat/completions`），确认远端命中与鉴权生效。
- 2025-10-21T11:40+09:00 — 图文消息验证返回文本提示，当前 `gpt-4o-mini` 不具备图像处理能力。
- 2025-10-21T11:45+09:00 — 重新运行实验使用远端字幕；产物位于 `outputs\experiments\EXP_20251021-114330\`；指标 `rank1=0.35`、`mAP=0.4973`；Gate PASS。
- 2025-10-21T11:50+09:00 — 验证 `captions_query.raw.jsonl`：端点为 `https://aizex.top/v1/chat/completions`，`status=200` 且 20/20 文本非空。