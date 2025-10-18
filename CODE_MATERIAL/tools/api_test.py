import os, json, re
from iso_api_subset_eval import call_caption_api

WORKDIR = os.getcwd()
img = os.path.join(WORKDIR, 'iso_staging', 'query', 'q_0000.jpg')

DEFAULT_PROMPT = (
    "Output concise English attribute phrases: top color/style, bottom color/style, shoes color/style, bag presence/type, salient accessories. Return comma-separated short phrases only."
)
PROMPT = os.environ.get('CAPTION_HEALTH_PROMPT', DEFAULT_PROMPT)

print('URL=', os.environ.get('CAPTION_API_URL'))
print('KEY set=', bool(os.environ.get('CAPTION_API_KEY')))
print('MODEL=', os.environ.get('CAPTION_API_MODEL'))
print('PROMPT=', PROMPT)
print('IMG=', img)

res = call_caption_api(img, PROMPT, os.environ.get('CAPTION_API_MODEL'))
text = res.get('text') or ''

# English + comma-separated diagnostics
has_alpha = bool(re.search(r"[A-Za-z]", text))
has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
english_ok = has_alpha and not has_cjk
parts = [p.strip() for p in text.replace('；', ', ').replace('，', ', ').split(',') if p.strip()]
comma_tokens_ge3 = len(parts) >= 3

print('ENGLISH_OK=', english_ok)
print('TOKENS>=3=', comma_tokens_ge3)
print('TEXT=', text)

out = {
    'image': os.path.relpath(img, WORKDIR),
    'prompt': PROMPT,
    'model': os.environ.get('CAPTION_API_MODEL'),
    'english_ok': english_ok,
    'comma_tokens_ge3': comma_tokens_ge3,
    'tokens': parts,
    'raw': res
}
print(json.dumps(out, ensure_ascii=False))