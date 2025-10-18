@echo off
set CFG=%1
if "%CFG%"=="" set CFG=configs\reid.yaml
set OUT=%2
if "%OUT%"=="" set OUT=outputs\ablation
if "%DATASET_ROOT%"=="" (
  for /f "tokens=2 delims=:" %%a in ('findstr /n "  root:" %CFG%') do set DATASET_ROOT=%%a
)
if "%DATASET_ROOT%"=="" ( echo DATASET_ROOT not set & exit /b 1 )
mkdir %OUT% 2>nul
echo text_backend,image_backend,rank1,mAP,n_query,n_gallery,elapsed_s > %OUT%\metrics.csv

for %%C in (clip_l14,clip_l14 clip_l14,clip_b16 clip_b16,clip_l14 clip_b16,clip_b16) do (
  for /f "tokens=1,2 delims=," %%t in ("%%C") do (
    set TB=%%t
    set IB=%%u
    set RUN=%OUT%\%%t_%%u
    mkdir !RUN! 2>nul
    setlocal enabledelayedexpansion
    python tools\captioner.py --root "%DATASET_ROOT%" --out "!RUN!\captions.json" --mode json
    python tools\embed_text.py --captions "!RUN!\captions.json" --out "!RUN!\text_embeds.npy"
    python tools\embed_image.py --root "%DATASET_ROOT%" --out "!RUN!\img_embeds.npy"
    python tools\retrieve_eval.py --text "!RUN!\text_embeds.npy" --img "!RUN!\img_embeds.npy" --out "!RUN!\metrics.json"
    for /f %%v in ('python - <<^"PY^" import json,sys;print(json.load(open(sys.argv[1]))['rank1']) ^"PY^" "!RUN!\metrics.json"') do set R1=%%v
    for /f %%v in ('python - <<^"PY^" import json,sys;print(json.load(open(sys.argv[1]))['mAP']) ^"PY^" "!RUN!\metrics.json"') do set MAP=%%v
    for /f %%v in ('python - <<^"PY^" import json,sys;print(json.load(open(sys.argv[1]))['n_query']) ^"PY^" "!RUN!\metrics.json"') do set NQ=%%v
    for /f %%v in ('python - <<^"PY^" import json,sys;print(json.load(open(sys.argv[1]))['n_gallery']) ^"PY^" "!RUN!\metrics.json"') do set NG=%%v
    echo !TB!,!IB!,!R1!,!MAP!,!NQ!,!NG!,[TBD] >> "%OUT%\metrics.csv"
    endlocal
  )
)
echo [ABLATE] wrote %OUT%\metrics.csv