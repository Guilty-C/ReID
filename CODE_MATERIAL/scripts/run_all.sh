#!/usr/bin/env bash
set -e
CFG=${1:-configs/reid.yaml}
OUT=outputs
ROOT=$(grep -E "^  root:" -n "$CFG" | awk '{print $2}' | tr -d '"')
mkdir -p "$OUT"/{logs,embeds,metrics}

python tools/captioner.py --root "$ROOT" --out "$OUT/captions.json" --mode json
python tools/embed_text.py --captions "$OUT/captions.json" --out "$OUT/embeds/text_embeds.npy"
python tools/embed_image.py --root "$ROOT" --out "$OUT/embeds/img_embeds.npy"
python tools/retrieve_eval.py --text "$OUT/embeds/text_embeds.npy" --img "$OUT/embeds/img_embeds.npy" --out "$OUT/metrics/metrics.json"