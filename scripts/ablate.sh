#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/reid.yaml}
ROOT=${2:-${DATASET_ROOT:-}}
OUT=${3:-outputs/ablation}
[[ -z "$ROOT" ]] && ROOT=$(grep -E "^  root:" -n "$CFG" | awk '{print $2}' | tr -d '"')
[[ -z "$ROOT" ]] && { echo "DATASET_ROOT not set"; exit 1; }

mkdir -p "$OUT"
echo "text_backend,image_backend,rank1,mAP,n_query,n_gallery,elapsed_s" > "$OUT/metrics.csv"

combos=("clip_l14,clip_l14" "clip_l14,clip_b16" "clip_b16,clip_l14" "clip_b16,clip_b16")
for combo in "${combos[@]}"; do
  tb=${combo%,*}; ib=${combo#*,}
  run_dir="$OUT/${tb}_${ib}"
  mkdir -p "$run_dir"
  SECONDS=0
  python tools/captioner.py --root "$ROOT" --out "$run_dir/captions.json" --mode json
  python tools/embed_text.py --captions "$run_dir/captions.json" --out "$run_dir/text_embeds.npy"
  python tools/embed_image.py --root "$ROOT" --out "$run_dir/img_embeds.npy"
  python tools/retrieve_eval.py --text "$run_dir/text_embeds.npy" --img "$run_dir/img_embeds.npy" --out "$run_dir/metrics.json"
  dur=$SECONDS
  rank1=$(python - <<'PY'\nimport json,sys;print(json.load(open(sys.argv[1]))['rank1'])\nPY "$run_dir/metrics.json")
  mapv=$(python - <<'PY'\nimport json,sys;print(json.load(open(sys.argv[1]))['mAP'])\nPY "$run_dir/metrics.json")
  nq=$(python - <<'PY'\nimport json,sys;print(json.load(open(sys.argv[1]))['n_query'])\nPY "$run_dir/metrics.json")
  ng=$(python - <<'PY'\nimport json,sys;print(json.load(open(sys.argv[1]))['n_gallery'])\nPY "$run_dir/metrics.json")
  echo "$tb,$ib,$rank1,$mapv,$nq,$ng,$dur" >> "$OUT/metrics.csv"
done
echo "[ABLATE] wrote $OUT/metrics.csv"