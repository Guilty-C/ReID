#!/usr/bin/env python
import os, csv, json, time, base64, hashlib, uuid, datetime
from urllib import request, error
from urllib.parse import urlparse

WORKDIR = os.getcwd()
FIXED_PROMPT = (
    "Return one JSON object with exactly these keys: top_color, top_style, bottom_color, bottom_style, shoes_color, shoes_style, bag (one of: none, backpack, handbag, shoulder, waist, unknown), accessories (array of short tokens). English only. No sentences, no extra keys. If uncertain use \"unknown\". Output a single JSON object only."
)
ISO_DIR = os.path.join(WORKDIR, 'iso')
ISO_STAGING_Q = os.path.join(WORKDIR, 'iso_staging', 'query')
ISO_STAGING_G = os.path.join(WORKDIR, 'iso_staging', 'gallery')
OUT_EMBEDS = os.path.join(WORKDIR, 'outputs', 'embeds')
OUT_METRICS = os.path.join(WORKDIR, 'outputs', 'metrics')
OUT_DIAG = os.path.join(WORKDIR, 'outputs', 'diagnostics')

# Helpers

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def l2_normalize_rows(arr):
    import numpy as np
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def read_manifest(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def file_exists(p):
    try:
        return os.path.isfile(p)
    except Exception:
        return False


def select_subset(manifest_rows, max_ids=100):
    # Group by id
    by_id = {}
    for r in manifest_rows:
        _id = r.get('id') or r.get('ID') or r.get('person_id')
        img = r.get('image_path') or r.get('path') or r.get('img')
        if not _id or not img:
            continue
        # try resolve relative to WORKDIR
        rp = os.path.join(WORKDIR, img)
        if file_exists(img):
            resolved = img
        elif file_exists(rp):
            resolved = rp
        else:
            # not found; skip
            continue
        by_id.setdefault(_id, []).append(resolved)

    # Filter ids with >=2 images, sort images deterministically
    eligible = []
    for _id, imgs in by_id.items():
        if len(imgs) >= 2:
            eligible.append((_id, sorted(imgs)))
    eligible.sort(key=lambda x: x[0])

    # Pick up to max_ids
    chosen = eligible[:max_ids]

    # For each id, choose 1 for query (first) and 1 for gallery (second)
    query_imgs, gallery_imgs, chosen_ids = [], [], []
    for _id, imgs in chosen:
        query_imgs.append(imgs[0])
        gallery_imgs.append(imgs[1])
        chosen_ids.append(_id)
    return query_imgs, gallery_imgs, chosen_ids


def select_subset_mock_market(max_ids=50):
    """Fallback: use data/mock_market where files are available.
    Treat base name without extension (e.g., mock_000) as ID; require presence in both splits.
    """
    root = os.path.join(WORKDIR, 'data', 'mock_market')
    q_dir = os.path.join(root, 'query')
    g_dir = os.path.join(root, 'bounding_box_test')
    if not os.path.isdir(q_dir) or not os.path.isdir(g_dir):
        return [], [], []
    q_files = [f for f in os.listdir(q_dir) if f.lower().endswith('.jpg')]
    g_files = [f for f in os.listdir(g_dir) if f.lower().endswith('.jpg')]
    # derive IDs by base names like mock_000 (strip extension)
    def base_id(name):
        return os.path.splitext(name)[0]
    q_ids = {base_id(f): f for f in q_files}
    g_ids = {}
    for f in g_files:
        bid = base_id(f)
        g_ids.setdefault(bid, []).append(f)
    # intersection IDs present in both
    ids = sorted(set(q_ids.keys()) & set(g_ids.keys()))
    ids = ids[:max_ids]
    query_imgs = []
    gallery_imgs = []
    chosen_ids = []
    for bid in ids:
        q_path = os.path.join(q_dir, q_ids[bid])
        g_list = g_ids.get(bid, [])
        if not g_list:
            continue
        g_path = os.path.join(g_dir, sorted(g_list)[0])
        query_imgs.append(q_path)
        gallery_imgs.append(g_path)
        chosen_ids.append(bid)
    return query_imgs, gallery_imgs, chosen_ids


def materialize_subset(query_imgs, gallery_imgs):
    # Copy files into iso_staging/query and iso_staging/gallery preserving order deterministically
    import shutil
    safe_mkdir(ISO_STAGING_Q)
    safe_mkdir(ISO_STAGING_G)

    q_paths_out = []
    g_paths_out = []

    for i, src in enumerate(query_imgs):
        ext = os.path.splitext(src)[1].lower() or '.jpg'
        dst = os.path.join(ISO_STAGING_Q, f'q_{i:04d}{ext}')
        shutil.copy2(src, dst)
        q_paths_out.append(dst)

    for i, src in enumerate(gallery_imgs):
        ext = os.path.splitext(src)[1].lower() or '.jpg'
        dst = os.path.join(ISO_STAGING_G, f'g_{i:04d}{ext}')
        shutil.copy2(src, dst)
        g_paths_out.append(dst)

    # Write iso file lists
    safe_mkdir(ISO_DIR)
    with open(os.path.join(ISO_DIR, 'query.txt'), 'w', encoding='utf-8') as f:
        for p in q_paths_out:
            f.write(os.path.relpath(p, WORKDIR) + '\n')
    with open(os.path.join(ISO_DIR, 'gallery.txt'), 'w', encoding='utf-8') as f:
        for p in g_paths_out:
            f.write(os.path.relpath(p, WORKDIR) + '\n')

    return q_paths_out, g_paths_out

# Re-add: base64 helper and caption API client

def b64_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def call_caption_api(image_path, prompt, model):
    url = os.environ.get('CAPTION_API_URL')
    api_key = os.environ.get('CAPTION_API_KEY')
    if api_key and api_key.startswith('api:'):
        api_key = api_key.split(':',1)[1]
    if not url or not api_key:
        return {'error': 'MISSING_ENV', 'text': '[API_ERROR] missing env', 'endpoint': '', 'status': None}

    b64 = b64_image(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/jpeg' if ext not in ['.png', '.webp'] else ('image/png' if ext=='.png' else 'image/webp')
    data_url = f'data:{mime};base64,{b64}'

    base = url.rstrip('/')
    # Build candidate endpoints: root, /v1/chat/completions, /v1/messages
    if ('/chat/completions' in base) or ('/responses' in base) or ('/messages' in base):
        candidates = [base]
    else:
        candidates = [base, base + '/v1/chat/completions', base + '/v1/messages']

    use_model = model or os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'
    chat_payload = {
        'model': use_model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': data_url, 'detail': 'low'}}
                ]
            }
        ]
    }
    messages_payload = {
        'model': use_model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': prompt},
                    {'type': 'input_image', 'image': b64}
                ]
            }
        ]
    }
    restful_payload = {'prompt': prompt, 'image_base64': b64, 'image_name': os.path.basename(image_path), 'model': use_model}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'} if api_key else {'Content-Type': 'application/json'}

    import requests, json
    last_err = ''
    last_status = None
    for endpoint in candidates:
        try:
            if '/chat/completions' in endpoint:
                r = requests.post(endpoint, headers=headers, json=chat_payload, timeout=60)
                status = r.status_code
                last_status = status
                data = r.json()
                # OpenAI-compatible extraction
                choice = (data.get('choices') or [{}])[0]
                msg = choice.get('message', {})
                content = msg.get('content')
                if isinstance(content, list):
                    text = ''.join([c.get('text','') for c in content])
                else:
                    text = (content or choice.get('text','') or '')
            elif '/messages' in endpoint:
                r = requests.post(endpoint, headers=headers, json=messages_payload, timeout=60)
                status = r.status_code
                last_status = status
                data = r.json()
                text = data.get('output',{}).get('choices',[{}])[0].get('content',[{}])[0].get('text','')
            else:
                r = requests.post(endpoint, headers=headers, json=restful_payload, timeout=60)
                status = r.status_code
                last_status = status
                data = r.json()
                text = data.get('text','') or data.get('content','') or ''
            if text:
                return {'text': text, 'endpoint': endpoint, 'status': status}
            last_err = f'EMPTY_TEXT_{endpoint} status={status}'
        except Exception as e:
            last_err = f'{e} endpoint={endpoint}'
            continue
    return {'error': last_err or 'REQUEST_FAILED', 'text': '', 'endpoint': candidates[-1] if candidates else '', 'status': last_status}


# New: emit aligned ID arrays and mapping CSV

def emit_ids_aligned(q_paths_out, g_paths_out, chosen_ids):
    import numpy as np
    out_dir = os.path.join(WORKDIR, 'outputs', 'indices')
    safe_mkdir(out_dir)
    q_ids_path = os.path.join(out_dir, 'query_ids.npy')
    g_ids_path = os.path.join(out_dir, 'gallery_ids.npy')
    ids_map_path = os.path.join(out_dir, 'ids_map.csv')

    np.save(q_ids_path, np.array(chosen_ids, dtype=object))
    np.save(g_ids_path, np.array(chosen_ids, dtype=object))

    with open(ids_map_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'id', 'split'])
        for p, _id in zip(q_paths_out, chosen_ids):
            w.writerow([os.path.relpath(p, WORKDIR), _id, 'query'])
        for p, _id in zip(g_paths_out, chosen_ids):
            w.writerow([os.path.relpath(p, WORKDIR), _id, 'gallery'])

    return q_ids_path, g_ids_path, ids_map_path

# Vocab and sanitize helpers for strict English JSON schema

ALLOWED_COLORS = {
    'black','white','gray','red','blue','green','yellow','brown','pink','purple','orange','beige'
}
ALLOWED_TOP_STYLES = {
    't-shirt','shirt','sweater','jacket','coat','hoodie','blouse','vest','cardigan'
}
ALLOWED_BOTTOM_STYLES = {
    'jeans','pants','trousers','shorts','skirt','leggings'
}
ALLOWED_SHOES_STYLES = {
    'sneakers','boots','heels','flats','sandals','loafers'
}
ALLOWED_BAG = {'none','backpack','handbag','shoulder','waist','unknown'}
ALLOWED_ACCESSORIES = {
    'hat','cap','glasses','sunglasses','scarf','belt','watch','bracelet','necklace','mask'
}


def _lower(s):
    return (s or '').strip().lower()


def sanitize_schema(obj):
    # ensure keys and values, default to 'unknown'
    o = {
        'top_color': _lower(obj.get('top_color')) if isinstance(obj.get('top_color'), str) else 'unknown',
        'top_style': _lower(obj.get('top_style')) if isinstance(obj.get('top_style'), str) else 'unknown',
        'bottom_color': _lower(obj.get('bottom_color')) if isinstance(obj.get('bottom_color'), str) else 'unknown',
        'bottom_style': _lower(obj.get('bottom_style')) if isinstance(obj.get('bottom_style'), str) else 'unknown',
        'shoes_color': _lower(obj.get('shoes_color')) if isinstance(obj.get('shoes_color'), str) else 'unknown',
        'shoes_style': _lower(obj.get('shoes_style')) if isinstance(obj.get('shoes_style'), str) else 'unknown',
        'bag': _lower(obj.get('bag')) if isinstance(obj.get('bag'), str) else 'unknown',
        'accessories': obj.get('accessories') if isinstance(obj.get('accessories'), list) else []
    }
    # map synonyms
    synonyms = {
        'grey': 'gray', 'blond': 'yellow', 'navy': 'blue', 'denim': 'jeans', 'sneaker': 'sneakers',
        'tee': 't-shirt', 'trouser': 'pants', 'slacks': 'pants', 'trainers': 'sneakers'
    }
    for k in ['top_color','bottom_color','shoes_color']:
        v = o[k]
        v = synonyms.get(v, v)
        o[k] = v if v in ALLOWED_COLORS else 'unknown'
    o['top_style'] = synonyms.get(o['top_style'], o['top_style'])
    o['bottom_style'] = synonyms.get(o['bottom_style'], o['bottom_style'])
    o['shoes_style'] = synonyms.get(o['shoes_style'], o['shoes_style'])
    o['top_style'] = o['top_style'] if o['top_style'] in ALLOWED_TOP_STYLES else 'unknown'
    o['bottom_style'] = o['bottom_style'] if o['bottom_style'] in ALLOWED_BOTTOM_STYLES else 'unknown'
    o['shoes_style'] = o['shoes_style'] if o['shoes_style'] in ALLOWED_SHOES_STYLES else 'unknown'
    o['bag'] = o['bag'] if o['bag'] in ALLOWED_BAG else 'unknown'
    acc = []
    for a in o['accessories']:
        aa = _lower(a)
        if aa in ALLOWED_ACCESSORIES:
            acc.append(aa)
    o['accessories'] = acc
    return o


def collapse_phrase(o):
    parts = []
    if o['top_color'] != 'unknown' or o['top_style'] != 'unknown':
        parts.append(' '.join([p for p in [o['top_color'], o['top_style']] if p != 'unknown']))
    if o['bottom_color'] != 'unknown' or o['bottom_style'] != 'unknown':
        parts.append(' '.join([p for p in [o['bottom_color'], o['bottom_style']] if p != 'unknown']))
    if o['shoes_color'] != 'unknown' or o['shoes_style'] != 'unknown':
        parts.append(' '.join([p for p in [o['shoes_color'], o['shoes_style']] if p != 'unknown']))
    if o['bag'] != 'unknown':
        parts.append(o['bag'] + ' bag')
    parts.extend([a for a in o['accessories']])
    # ensure at least 3 tokens
    tokens = ', '.join(parts)
    if len(tokens.replace(',', ' ').split()) < 3:
        tokens = ', '.join(parts + ['unknown'])
    return tokens

# Strict JSON schema captions with sanitize and persistence

def ensure_captions_schema(query_paths):
    captions_dir = os.path.join(WORKDIR, 'outputs', 'captions')
    diag_dir = os.path.join(WORKDIR, 'outputs', 'diagnostics')
    safe_mkdir(captions_dir)
    safe_mkdir(diag_dir)

    raw_log_path = os.path.join(diag_dir, 'api_raw_schema.jsonl')
    json_out = os.path.join(captions_dir, 'iso_api_schema.json')
    json_sanitized_out = os.path.join(captions_dir, 'iso_api_schema_sanitized.json')
    phrases_out = os.path.join(captions_dir, 'iso_api_schema_phrases.json')

    ordered_json = {}
    ordered_sanitized = {}
    phrase_map = {}

    model = os.environ.get('CAPTION_API_MODEL')
    raw_f = open(raw_log_path, 'a', encoding='utf-8')

    fallback_count = 0
    for qp in query_paths:
        key = os.path.relpath(qp, WORKDIR)
        # bounded retries
        retries = 3
        delay = 1.0
        parsed_obj = None
        last_err = None
        for t in range(retries):
            res = call_caption_api(qp, FIXED_PROMPT, model)
            raw_entry = {
                'image_path': key,
                'ts': int(time.time()),
                'error': res.get('error'),
            }
            if 'raw_obj' in res:
                raw_entry['raw_obj'] = res['raw_obj']
            raw_f.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')

            txt = res.get('text')
            if txt and not txt.startswith('[API_ERROR]'):
                # try to parse JSON object from text
                candidate = txt
                if '```' in candidate:
                    # extract between first '{' and last '}'
                    s = candidate.find('{')
                    e = candidate.rfind('}')
                    if s != -1 and e != -1 and e > s:
                        candidate = candidate[s:e+1]
                try:
                    obj = json.loads(candidate)
                    # verify keys
                    keys = set(obj.keys())
                    required = {'top_color','top_style','bottom_color','bottom_style','shoes_color','shoes_style','bag','accessories'}
                    if required.issubset(keys):
                        parsed_obj = obj
                        break
                    else:
                        last_err = 'SCHEMA_KEYS'
                except Exception:
                    last_err = 'PARSE_JSON'
            else:
                last_err = res.get('error')
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        if not parsed_obj:
            parsed_obj = {
                'top_color': 'unknown','top_style': 'unknown',
                'bottom_color': 'unknown','bottom_style': 'unknown',
                'shoes_color': 'unknown','shoes_style': 'unknown',
                'bag': 'unknown','accessories': []
            }
            fallback_count += 1
        sanitized = sanitize_schema(parsed_obj)
        phrase = collapse_phrase(sanitized)
        ordered_json[key] = parsed_obj
        ordered_sanitized[key] = sanitized
        phrase_map[key] = [phrase]

    raw_f.close()

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(ordered_json, f, ensure_ascii=False, indent=2)
    with open(json_sanitized_out, 'w', encoding='utf-8') as f:
        json.dump(ordered_sanitized, f, ensure_ascii=False, indent=2)
    with open(phrases_out, 'w', encoding='utf-8') as f:
        json.dump(phrase_map, f, ensure_ascii=False, indent=2)

    # index CSV for inspection
    index_csv = os.path.join(diag_dir, 'iso_prompt_schema_index.csv')
    with open(index_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path','phrase'])
        for k, v in phrase_map.items():
            w.writerow([k, v[0]])

    # meta
    meta_path = os.path.join(diag_dir, 'iso_prompt_schema_meta.json')
    meta = {
        'fallback_count': fallback_count,
        'total': len(query_paths),
        'fallback_rate': round(fallback_count / max(1, len(query_paths)), 4)
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return phrases_out, json_out, json_sanitized_out, index_csv, meta_path


def run_cmd(cmd):
    import subprocess
    print('[CMD]', cmd)
    cp = subprocess.run(cmd, shell=True)
    if cp.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def normalize_npy(in_path, out_path):
    import numpy as np
    arr = np.load(in_path)
    arr = l2_normalize_rows(arr)
    np.save(out_path, arr)
    print(f'[OK] normalized {in_path} -> {out_path}: {arr.shape}')
    return arr.shape


def main():
    safe_mkdir(ISO_DIR)
    safe_mkdir(ISO_STAGING_Q)
    safe_mkdir(ISO_STAGING_G)
    safe_mkdir(OUT_EMBEDS)
    safe_mkdir(OUT_METRICS)
    safe_mkdir(OUT_DIAG)

    manifest_path = os.path.join(WORKDIR, 'manifest.csv')
    rows = read_manifest(manifest_path)
    q_imgs, g_imgs, ids = select_subset(rows, max_ids=100)
    if len(q_imgs) == 0 or len(g_imgs) == 0:
        print('[WARN] No valid images from manifest.csv. Falling back to data/mock_market subset.')
        q_imgs, g_imgs, ids = select_subset_mock_market(max_ids=50)
    q_paths, g_paths = materialize_subset(q_imgs, g_imgs)

    # Emit aligned IDs
    q_ids_path, g_ids_path, ids_map_path = emit_ids_aligned(q_paths, g_paths, ids)

    # Captions with strict schema and sanitize
    phrases_path, json_out, json_sanitized_out, index_csv, meta_path = ensure_captions_schema(q_paths)

    # Embeddings: text from sanitized phrases, image reuse if present
    text_out = os.path.join(OUT_EMBEDS, 'text_iso_schema.npy')
    text_norm_out = os.path.join(OUT_EMBEDS, 'text_iso_schema_norm.npy')
    img_out = os.path.join(OUT_EMBEDS, 'img_iso_real.npy')

    run_cmd(f'python tools/embed_text.py --captions "{phrases_path}" --out "{text_out}" --backend mock')

    if not file_exists(img_out):
        run_cmd(f'python tools/embed_image.py --root "{os.path.join(WORKDIR, "iso_staging")}" --split gallery --out "{img_out}" --backend mock')

    # Normalize
    normalize_npy(text_out, text_norm_out)

    # Evaluate ID-aware
    metrics_out = os.path.join(OUT_METRICS, 'metrics_iso_schema.json')
    run_cmd(f'python tools/retrieve_eval.py --text "{text_norm_out}" --img "{img_out}" --out "{metrics_out}" --query-ids "{q_ids_path}" --gallery-ids "{g_ids_path}"')

    # Load metrics
    with open(metrics_out, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # Append ablation row
    ablation_csv = os.path.join(WORKDIR, 'outputs', 'ablation', 'iso_prompt_metrics.csv')
    safe_mkdir(os.path.dirname(ablation_csv))
    header = ['prompt','rank1','mAP','n_query','n_gallery']
    row = ['schema', f"{metrics.get('rank1',0.0):.6f}", f"{metrics.get('mAP',0.0):.6f}", str(metrics.get('n_query',0)), str(metrics.get('n_gallery',0))]
    if not os.path.isfile(ablation_csv):
        with open(ablation_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow(header); w.writerow(row)
    else:
        with open(ablation_csv, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow(row)

    # History CSV
    history_csv = os.path.join(OUT_METRICS, 'history.csv')
    with open(history_csv, 'a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([datetime.datetime.utcnow().isoformat()+'Z','ISO','schema','mock', metrics.get('rank1',0.0), metrics.get('mAP',0.0), metrics.get('n_query',0), metrics.get('n_gallery',0)])

    # Update docs
    docs_path = os.path.join(WORKDIR, 'docs', 'Execution_Summary.md')
    safe_mkdir(os.path.dirname(docs_path))
    with open(docs_path, 'a', encoding='utf-8') as f:
        f.write('\n\n### Evaluator Correction (ID-aware)\n')
        f.write('- Uses full Nq×Ng similarity and ID arrays for Rank-1/mAP.\n')
        f.write(f"- Metrics (schema): Rank-1={metrics.get('rank1',0.0):.4f}, mAP={metrics.get('mAP',0.0):.4f}, n_query={metrics.get('n_query',0)}, n_gallery={metrics.get('n_gallery',0)}\n")
        f.write('- Captions enforce strict English JSON schema; sanitize to a small vocabulary.\n')
        f.write('- Artifacts: captions JSON(raw/sanitized), phrases, aligned IDs, metrics JSON.\n')

    print('[OK] ID arrays + schema captions + eval completed')

if __name__ == '__main__':
    main()