@echo off
set CFG=configs\reid.yaml
set OUT=outputs
for /f "tokens=2 delims=:" %%a in ('findstr /n "  root:" %CFG%') do set ROOT=%%a
mkdir %OUT%\logs %OUT%\embeds %OUT%\metrics 2>nul
python tools\captioner.py --root %ROOT% --out %OUT%\captions.json --mode json
python tools\embed_text.py --captions %OUT%\captions.json --out %OUT%\embeds\text_embeds.npy
python tools\embed_image.py --root %ROOT% --out %OUT%\embeds\img_embeds.npy
python tools\retrieve_eval.py --text %OUT%\embeds\text_embeds.npy --img %OUT%\embeds\img_embeds.npy --out %OUT%\metrics\metrics.json