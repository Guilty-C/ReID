# RUNBOOK — PRP-ReID Phase 5–6
## 先决
- 确认存在：configs/reid.yaml、tools/{captioner.py,embed_text.py,embed_image.py,retrieve_eval.py}、scripts/run_all.{sh,bat}
- 设置：dataset.root= "<DATASET_ROOT>"；输出目录 `V2/outputs`

## Linux
```bash
cd "V2"
python -m venv .venv && source .venv/bin/activate
pip install -U pip && (test -f requirements.txt && pip install -r requirements.txt || true)
ts=$(date +%Y%m%d-%H%M); mkdir -p ARTIFACTS/run_logs/$ts
{ time bash scripts/run_all.sh --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --out "outputs"; } \
  2>&1 | tee ARTIFACTS/run_logs/$ts/run.log
```

## Windows (PowerShell)

```powershell
cd "V2"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip; if (Test-Path requirements.txt) { pip install -r requirements.txt }
$ts = Get-Date -Format "yyyyMMdd-HHmm"; New-Item -ItemType Directory -Force -Path "ARTIFACTS/run_logs/$ts" | Out-Null
Measure-Command { .\scripts\run_all.bat --cfg configs\reid.yaml --dataset-root "<DATASET_ROOT>" --out "outputs" } `
  | Out-String | Tee-Object -FilePath "ARTIFACTS/run_logs/$ts/run.log" -Append
```

## 期望产物

* outputs/{captions.json,text_embeds.npy,img_embeds.npy,metrics/metrics.json}
* ARTIFACTS/run_logs/<timestamp>/run.log

## 自检

* `np.load` 检查 `*_embeds.npy` 形状>0
* `metrics.json` 必含：rank1,mAP,n_query,n_gallery，均为数值
* 失败情形与替代：

  * 无 GPU/LLM Key：照常运行，占位写入，标注 `[TBD]`
  * 数据路径无效：创建 mock 子集 `V2/data/mock_market/{query,bounding_box_test}` 并重跑

## 本地测试与消融

```bash
# Linux
pytest -q
python tools/smoke_test.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --out outputs
bash scripts/ablate.sh configs/reid.yaml "<DATASET_ROOT>" outputs/ablation

# Windows (PowerShell)
pytest -q
python tools/smoke_test.py --cfg configs/reid.yaml --dataset-root "<DATASET_ROOT>" --out outputs
scripts\ablate.bat configs\reid.yaml "<DATASET_ROOT>" outputs\ablation
```

## 打包命令

```bash
# Linux
ts=$(date +%Y%m%d-HH%M)
mkdir -p deliverables_$ts
cp -r docs tests .github scripts outputs ARTIFACTS deliverables_$ts/
rm -f deliverables_$ts/outputs/**/*.jpg 2>/dev/null || true
zip -r deliverables_$ts.zip deliverables_$ts -x "*.jpg" "*.png" "*.bmp"

# Windows (PowerShell)
$ts = Get-Date -Format "yyyyMMdd-HHmm"
Compress-Archive -Path docs, tests, .github, scripts, outputs, ARTIFACTS -DestinationPath "deliverables_$ts.zip" -Force