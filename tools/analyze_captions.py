import json, os, re, csv
from glob import glob

CAP_DIR = os.path.join('outputs', 'captions')
OUT_DIR = os.path.join('outputs', 'ablation')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'caption_quality.csv')

is_chinese = re.compile(r'[\u4e00-\u9fff]')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_one(path):
    data = load_json(path)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        vals = []
        for v in data.values():
            if isinstance(v, list):
                vals.extend(v)
            elif isinstance(v, dict) and 'text' in v:
                vals.append(v['text'])
            elif isinstance(v, str):
                vals.append(v)
        items = vals
    else:
        items = []
    total = len(items)
    chinese = 0
    api_err = 0
    lang_mismatch = 0
    empty = 0
    for it in items:
        txt = it.get('text') if isinstance(it, dict) else str(it)
        if not txt:
            empty += 1
            continue
        if txt.startswith('[API_ERROR]'):
            api_err += 1
        if 'language mismatch' in txt or 'TBD' in txt:
            lang_mismatch += 1
        if is_chinese.search(txt):
            chinese += 1
    ratio = (chinese / total) if total else 0.0
    return {
        'file': os.path.basename(path),
        'total': total,
        'chinese': chinese,
        'chinese_ratio': round(ratio, 4),
        'api_error': api_err,
        'lang_mismatch': lang_mismatch,
        'empty': empty,
    }

def main():
    files = sorted(glob(os.path.join(CAP_DIR, 'iso_api_P*.json')))
    rows = [analyze_one(p) for p in files]
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['file','total','chinese','chinese_ratio','api_error','lang_mismatch','empty'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {OUT_CSV}')

if __name__ == '__main__':
    main()