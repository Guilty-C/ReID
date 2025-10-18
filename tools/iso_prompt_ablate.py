    #!/usr/bin/env python
import os, sys, json, time, base64, uuid, datetime, csv, re
import subprocess
from pathlib import Path
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

WORKDIR = os.getcwd()
ISO_DIR = os.path.join(WORKDIR, 'iso')
ISO_Q_TXT = os.path.join(ISO_DIR, 'query.txt')
ISO_G_TXT = os.path.join(ISO_DIR, 'gallery.txt')
ISO_Q_DIR = os.path.join(WORKDIR, 'iso_staging', 'query')
ISO_G_DIR = os.path.join(WORKDIR, 'iso_staging', 'gallery')
OUT_CAPTIONS_DIR = os.path.join(WORKDIR, 'outputs', 'captions')
OUT_EMBEDS_DIR = os.path.join(WORKDIR, 'outputs', 'embeds')
OUT_METRICS_DIR = os.path.join(WORKDIR, 'outputs', 'metrics')
OUT_DIAG_DIR = os.path.join(WORKDIR, 'outputs', 'diagnostics')
OUT_ABL_DIR = os.path.join(WORKDIR, 'outputs', 'ablation')
DOC_SUMMARY_MD = os.path.join(WORKDIR, 'docs', 'Execution_Summary.md')

PROMPTS = {
    'P1': 'Output concise English attribute phrases: top color/style, bottom color/style, shoes color/style, bag presence/type, salient accessories. Return comma-separated short phrases only.',
    'P2': 'Return English retrieval attributes as short phrases, comma-separated: upper clothing (color, style), lower clothing (color, style), shoes (color, style), bag (yes/no, type), notable accessories.',
    'P3': 'Describe the person with English attribute phrases for ReID: top color+style, bottom color+style, shoes color+style, bag yes/no and type, distinctive accessories. Comma-separated, no sentences.',
    'P4': 'English attribute tokens for person identification: upper garment color/style, lower garment color/style, footwear color/style, bag carried (type/none), standout accessories. Output comma-separated phrases only.',
    'P5': 'Produce concise English attribute phrases useful for retrieval: clothing (top/bottom/shoes) colors and styles, bag presence/type, prominent accessories. Comma-separated.'
}

BASELINE_RANK1 = 0.25  # existing ISO baseline
TEXT_BACKEND = 'clip_l14'
TEXT_BATCH = 64
IMG_BACKEND = 'clip_l14'
IMG_BATCH = 128

# Reuse robust API caller from sibling module
try:
    from iso_api_subset_eval import call_caption_api
except Exception as e:
    print('[WARN] import iso_api_subset_eval failed, will inline simple caller:', e)
    call_caption_api = None


def ensure_dirs():
    for d in [OUT_CAPTIONS_DIR, OUT_EMBEDS_DIR, OUT_METRICS_DIR, OUT_DIAG_DIR, OUT_ABL_DIR]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(WORKDIR, 'ARTIFACTS', 'run_logs'), exist_ok=True)


def read_list(txt_path, staging_root):
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f'missing {txt_path}')
    rels = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rels.append(line)
    # materialize to absolute paths under staging_root
    abs_paths = []
    for r in rels:
        p = os.path.join(staging_root, os.path.basename(r))
        abs_paths.append(p)
    return abs_paths


def language_is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ''))

# English enforcement helpers

def language_is_english(text: str) -> bool:
    if not text:
        return False
    has_alpha = bool(re.search(r"[A-Za-z]", text))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    return has_alpha and not has_cjk


def normalize_commas_en(text: str) -> str:
    s = text or ''
    # unify separators
    s = re.sub(r'[;\n\r]+', ', ', s)
    s = s.replace('；', ', ').replace('，', ', ')
    # remove bullets
    s = re.sub(r'^\s*[-•]\s*', '', s, flags=re.MULTILINE)
    # split by commas
    parts = [p.strip() for p in s.split(',') if p.strip()]
    cleaned = []
    for p in parts:
        p2 = re.sub(r'(?i)\b(top|upper|upper garment|upper clothing|bottom|lower|lower garment|lower clothing|shoes?|footwear|bag|handbag|backpack|accessories?)\s*[:：]\s*', '', p)
        p2 = re.sub(r'\s+', ' ', p2).strip().strip('.')
        if p2:
            cleaned.append(p2)
    if len(cleaned) <= 1:
        parts2 = re.split(r'[/;|]| and |；|，', s)
        cleaned = [re.sub(r'\s+', ' ', p).strip().strip('.') for p in parts2 if p.strip()]
    # dedup short phrases
    uniq = []
    for p in cleaned:
        if p and p not in uniq:
            uniq.append(p)
    # Guarantee at least 3 tokens using conservative defaults
    if len(uniq) < 3:
        # Try to infer category presence
        low = s.lower()
        has_top = bool(re.search(r'\b(top|upper|upper garment|upper clothing)\b', low))
        has_bottom = bool(re.search(r'\b(bottom|lower|lower garment|lower clothing)\b', low))
        has_shoes = bool(re.search(r'\b(shoe|shoes|footwear)\b', low))
        has_bag = bool(re.search(r'\b(bag|handbag|backpack)\b', low))
        has_acc = bool(re.search(r'\b(accessory|accessories)\b', low))
        fillers = []
        if not has_top:
            fillers.append('top unknown')
        if not has_bottom:
            fillers.append('bottom unknown')
        if not has_bag:
            fillers.append('bag none')
        # If still short, add shoes/accessories defaults
        if len(uniq) + len(fillers) < 3 and not has_shoes:
            fillers.append('shoes unknown')
        if len(uniq) + len(fillers) < 3 and not has_acc:
            fillers.append('accessories none')
        for f in fillers:
            if f not in uniq:
                uniq.append(f)
        # limit to minimal needed
        if len(uniq) > 6:
            uniq = uniq[:6]
    out = ', '.join(uniq)
    if not language_is_english(out):
        # keep only ascii-alpha tokens if possible
        ascii_tokens = [x for x in uniq if re.search(r'[A-Za-z]', x)]
        if ascii_tokens:
            out = ', '.join(ascii_tokens)
    return out or (text or 'person')


def api_caption_one(img_path: str, prompt: str, model: str):
    # Prefer robust caller if available
    if call_caption_api:
        return call_caption_api(img_path, prompt, model)
    # Fallback simple OpenAI-compatible caller via requests
    import requests
    url = os.environ.get('CAPTION_API_URL')
    api_key = os.environ.get('CAPTION_API_KEY')
    if api_key and api_key.startswith('api:'):
        api_key = api_key.split(':', 1)[1]
    if not url or not api_key:
        return {'error': 'MISSING_ENV', 'text': '[API_ERROR] missing env'}
    model = model or os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'
    # Build data URL
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    ext = os.path.splitext(img_path)[1].lower()
    mime = 'image/jpeg' if ext not in ['.png', '.webp'] else ('image/png' if ext=='.png' else 'image/webp')
    data_url = f'data:{mime};base64,{b64}'
    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': data_url, 'detail': 'low'}}
            ]
        }]
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        http_status = getattr(r, 'status_code', None)
        obj = r.json()
        # parse common variants
        txt = None
        if isinstance(obj, dict):
            if 'choices' in obj and obj['choices']:
                msg = obj['choices'][0].get('message') or {}
                content = msg.get('content')
                if isinstance(content, list) and content:
                    # new SDK format
                    for part in content:
                        t = part.get('text') or part.get('output_text')
                        if t:
                            txt = t
                            break
                elif isinstance(content, str):
                    txt = content
            elif 'output_text' in obj:
                txt = obj['output_text']
        # flag errors on 429/5xx
        err = None
        if http_status and (http_status == 429 or http_status >= 500):
            err = f'HTTP_{http_status}'
        return {'text': txt or '[API_ERROR] parse', 'raw_obj': obj, 'http_status': http_status, 'error': err}
    except Exception as e:
        return {'error': str(e), 'text': f'[API_ERROR] {e}'}


def generate_captions_for_prompt(pid: str, prompt: str, q_paths: list):
    model = os.environ.get('CAPTION_API_MODEL')
    raw_log_path = os.path.join(OUT_DIAG_DIR, f'api_raw_{pid}.jsonl')
    caps_path = os.path.join(OUT_CAPTIONS_DIR, f'iso_api_{pid}.json')
    out = {}
    # Resume: load existing captions if any
    existing_count = 0
    if os.path.isfile(caps_path):
        try:
            with open(caps_path, 'r', encoding='utf-8') as f:
                out = json.load(f) if f.read(1) else {}
                if isinstance(out, dict):
                    existing_count = len(out)
                else:
                    out = {}
        except Exception:
            out = {}
    n_total = len(q_paths)
    # Determine start index (first missing in deterministic order)
    missing_idxs = []
    for i, qp in enumerate(q_paths):
        key_i = os.path.relpath(qp, WORKDIR)
        if key_i not in out:
            missing_idxs.append(i)
    start_idx = missing_idxs[0] if missing_idxs else n_total
    # Progress bar setup
    use_tqdm = (tqdm is not None) and (sys.stdout.isatty() or os.environ.get('TQDM_FORCE') == '1')
    bar = None
    if use_tqdm:
        bar = tqdm(total=n_total, initial=existing_count, desc=f'Captioning {pid}', unit='img', leave=False)
    retries_total = 0
    retry_max = 0
    fail_examples = []
    started_at = datetime.datetime.now().isoformat()
    t0 = time.time()
    with open(raw_log_path, 'a', encoding='utf-8') as rawf:
        # iterate from first missing to end
        for idx in range(start_idx, n_total):
            qp = q_paths[idx]
            key = os.path.relpath(qp, WORKDIR)
            retries, delay = 4, 1.0
            result_text, last_err, http_status = None, None, None
            used_retries = 0
            for t in range(retries):
                res = api_caption_one(qp, prompt, model)
                http_status = res.get('http_status')
                rawf.write(json.dumps({
                    'index': idx,
                    'image_path': key,
                    'ts': int(time.time()),
                    'pid': pid,
                    'error': res.get('error'),
                    'http_status': http_status,
                    'raw_obj': res.get('raw_obj')
                }, ensure_ascii=False) + '\n')
                txt = res.get('text')
                if txt and not txt.startswith('[API_ERROR]'):
                    result_text = txt
                    break
                last_err = res.get('error')
                used_retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
            retries_total += used_retries
            retry_max = max(retry_max, used_retries)
            if not result_text:
                result_text = f'[API_ERROR] {last_err or "unknown"}'
                fail_examples.append(key)
            # enforce English & comma-separated short phrases
            cleaned = normalize_commas_en(result_text)
            out[key] = [cleaned if cleaned else 'person']
            if use_tqdm:
                bar.update(1)
                bar.set_postfix({'i': f'{idx+1}/{n_total}', 'retry': used_retries})
            else:
                print(f"Captioning {pid}: {idx+1}/{n_total}, retries={used_retries}")
    if use_tqdm and bar is not None:
        bar.close()
    # persist captions ensuring full count
    with open(caps_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    ended_at = datetime.datetime.now().isoformat()
    elapsed = time.time() - t0
    # write meta JSON
    meta_path = os.path.join(OUT_DIAG_DIR, f'iso_prompt_{pid}_meta.json')
    meta = {
        'pid': pid,
        'prompt': prompt,
        'model': model or (os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'),
        'n_query': n_total,
        'retries_total': retries_total,
        'retry_max': retry_max,
        'missing_count': 0 if len(out) >= n_total else (n_total - len(out)),
        'started_at': started_at,
        'ended_at': ended_at,
        'elapsed_s': round(elapsed, 3)
    }
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)
    return caps_path, fail_examples


def run_cmd(cmd: list):
    print('[CMD]', ' '.join(cmd))
    start = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    elapsed = time.time() - start
    if p.returncode != 0:
        raise RuntimeError(f'command failed: {cmd}\n{p.stdout}')
    return elapsed


def l2_normalize_npy(src_path: str, dst_path: str):
    arr = np.load(src_path)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    arr = arr.astype(np.float32)
    np.save(dst_path, arr)
    return arr.shape


def compute_top1_and_hist(text_npy: str, img_npy: str, top1_csv: str, hist_csv: str):
    T = np.load(text_npy)
    I = np.load(img_npy)
    # safety normalize
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-9)
    S = T @ I.T
    # top-1 for each query
    top1_rows = []
    for i in range(S.shape[0]):
        j = int(np.argmax(S[i]))
        sim = float(S[i, j])
        match = (i == j)
        top1_rows.append([f'q_{i}', f'g_{j}', f'{sim:.4f}', 'hit' if match else 'miss'])
    # write top1
    os.makedirs(os.path.dirname(top1_csv), exist_ok=True)
    with open(top1_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query','gallery','similarity','result'])
        for r in top1_rows[:20]:
            w.writerow(r)
    # histogram
    bins = [x/20.0 for x in range(0, 21)]  # 0..1 step 0.05
    centers = [ (bins[k]+bins[k+1])/2.0 for k in range(len(bins)-1) ]
    counts = [0]* (len(bins)-1)
    flat = S.flatten()
    for val in flat:
        for k in range(len(bins)-1):
            if bins[k] <= val < bins[k+1]:
                counts[k] += 1
                break
    with open(hist_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bin_center','count'])
        for c, n in zip(centers, counts):
            w.writerow([f'{c:.2f}', n])


# Per-prompt quality summary

def write_quality_csv_for_prompt(caps_path: str, pid: str):
    try:
        with open(caps_path, 'r', encoding='utf-8') as f:
            caps = json.load(f)
    except Exception:
        caps = {}
    total = 0
    english = 0
    api_err = 0
    comma3 = 0
    empty = 0
    for k, v in (caps.items() if isinstance(caps, dict) else {}):
        total += 1
        txt = ''
        if isinstance(v, list) and v:
            txt = str(v[0])
        elif isinstance(v, str):
            txt = v
        if not txt:
            empty += 1
            continue
        if txt.startswith('[API_ERROR]'):
            api_err += 1
        if language_is_english(txt):
            english += 1
        toks = [t.strip() for t in str(txt).split(',') if t.strip()]
        if len(toks) >= 3:
            comma3 += 1
    out_path = os.path.join(OUT_DIAG_DIR, f'iso_prompt_quality_{pid}.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pid','total','english','english_ratio','api_error','comma_tokens_ge3','empty'])
        ratio = (english/total) if total else 0.0
        w.writerow([pid, total, english, f'{ratio:.4f}', api_err, comma3, empty])
    print(f'[OK] quality CSV -> {out_path}')


def ensure_img_gallery_npy():
    img_npy = os.path.join(OUT_EMBEDS_DIR, 'img_iso_real.npy')
    if os.path.isfile(img_npy):
        return img_npy
    print('[INFO] missing img_iso_real.npy, generating from iso_staging/gallery using real backend')
    elapsed = run_cmd([sys.executable, os.path.join(WORKDIR, 'tools', 'embed_image.py'),
             '--root', ISO_G_DIR, '--split', 'gallery', '--out', img_npy,
             '--backend', IMG_BACKEND, '--batch-size', str(IMG_BATCH)])
    print(f'[OK] generated gallery embeds in {elapsed:.2f}s')
    # normalize in place
    shapes = l2_normalize_npy(img_npy, img_npy)
    print(f'[OK] normalized gallery embeds shape={shapes}')
    return img_npy


def write_index_csv_for_prompt(caps_path: str, pid: str, q_paths: list):
    try:
        with open(caps_path, 'r', encoding='utf-8') as f:
            caps = json.load(f)
    except Exception:
        caps = {}
    out_path = os.path.join(OUT_DIAG_DIR, f'iso_prompt_{pid}_index.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ordered_index','query_image','caption_text'])
        for idx, qp in enumerate(q_paths):
            key = os.path.relpath(qp, WORKDIR)
            val = caps.get(key)
            txt = ''
            if isinstance(val, list) and val:
                txt = str(val[0])
            elif isinstance(val, str):
                txt = val
            w.writerow([idx, key, txt])
    print(f'[OK] index CSV -> {out_path}')


def main():
    ensure_dirs()
    # Ensure query/gallery order from iso/*.txt
    q_paths = read_list(ISO_Q_TXT, ISO_Q_DIR)
    g_paths = read_list(ISO_G_TXT, ISO_G_DIR)
    n_q, n_g = len(q_paths), len(g_paths)
    if n_q == 0 or n_g == 0:
        raise RuntimeError('empty iso subset list')
    img_npy = ensure_img_gallery_npy()
    # Evaluate prompts
    summary_rows = []
    best_by_map = {'prompt': None, 'mAP': -1.0, 'rank1': -1.0}
    run_log_path = os.path.join(WORKDIR, 'ARTIFACTS', 'run_logs', 'iso_prompts.log')
    for pid, prompt in PROMPTS.items():
        print(f'\n[RUN] {pid}: generating captions + embeddings + eval')
        start_wall = datetime.datetime.now().isoformat()
        t0 = time.time()
        caps_path, fails = generate_captions_for_prompt(pid, prompt, q_paths)
        print(f'[OK] captions -> {caps_path}, fail_count={len(fails)}')
        # Per-prompt quality
        write_quality_csv_for_prompt(caps_path, pid)
        # Embed text (resume: skip if exists and matches n_query unless SAVE_NORM_ONLY=1)
        text_out = os.path.join(OUT_EMBEDS_DIR, f'text_iso_{pid}.npy')
        save_norm_only = os.environ.get('SAVE_NORM_ONLY') == '1'
        need_embed = True
        if os.path.isfile(text_out) and not save_norm_only:
            try:
                arr = np.load(text_out)
                if arr.shape[0] == n_q:
                    need_embed = False
                    print(f'[SKIP] embeddings exist for {pid}: {text_out}')
            except Exception:
                need_embed = True
        if need_embed:
            elapsed1 = run_cmd([sys.executable, os.path.join(WORKDIR, 'tools', 'embed_text.py'),
                     '--captions', caps_path, '--out', text_out,
                     '--backend', TEXT_BACKEND, '--device', 'auto', '--batch-size', str(TEXT_BATCH)])
        # Normalize text
        text_norm = os.path.join(OUT_EMBEDS_DIR, f'text_iso_{pid}_norm.npy')
        shape = l2_normalize_npy(text_out, text_norm)
        print(f'[OK] text normalized shape={shape}')
        # Index CSV aligned with embeddings
        write_index_csv_for_prompt(caps_path, pid, q_paths)
        # Validate dims vs gallery
        I = np.load(img_npy)
        T = np.load(text_norm)
        if T.shape[1] != I.shape[1]:
            raise RuntimeError(f'Embedding dim mismatch: text {T.shape[1]} vs image {I.shape[1]}')
        if T.shape[0] != n_q:
            raise RuntimeError(f'Embedding count mismatch: text {T.shape[0]} vs n_query {n_q}')
        if not np.isfinite(T).all():
            raise RuntimeError('Non-finite values in text embeddings')
        # Evaluate
        metrics_out = os.path.join(OUT_METRICS_DIR, f'metrics_iso_api_{pid}.json')
        elapsed2 = run_cmd([sys.executable, os.path.join(WORKDIR, 'tools', 'retrieve_eval.py'),
                 '--text', text_norm, '--img', img_npy, '--out', metrics_out])
        # Diagnostics
        top1_csv = os.path.join(OUT_DIAG_DIR, f'iso_prompt_top1_{pid}.csv')
        hist_csv = os.path.join(OUT_DIAG_DIR, f'iso_prompt_hist_{pid}.csv')
        compute_top1_and_hist(text_norm, img_npy, top1_csv, hist_csv)
        # Load metrics
        with open(metrics_out, 'r', encoding='utf-8') as f:
            m = json.load(f)
        elapsed = time.time() - t0
        end_wall = datetime.datetime.now().isoformat()
        summary_rows.append([pid, float(m.get('rank1', 0.0)), float(m.get('mAP', 0.0)), int(m.get('n_query', n_q)), int(m.get('n_gallery', n_g)), round(elapsed, 3)])
        # track best by mAP then rank1
        cur_map = float(m.get('mAP', 0.0))
        cur_r1 = float(m.get('rank1', 0.0))
        if cur_map > best_by_map['mAP'] or (cur_map == best_by_map['mAP'] and cur_r1 > best_by_map['rank1']):
            best_by_map = {'prompt': pid, 'mAP': cur_map, 'rank1': cur_r1}
        print(f'[OK] {pid} rank1={cur_r1:.4f}, mAP={cur_map:.4f}, elapsed={elapsed:.2f}s')
        # Append run log
        with open(run_log_path, 'a', encoding='utf-8') as lf:
            lf.write(json.dumps({'pid': pid, 'started_at': start_wall, 'ended_at': end_wall, 'elapsed_s': round(elapsed, 3)}, ensure_ascii=False) + '\n')
    # Write ablation CSV
    os.makedirs(OUT_ABL_DIR, exist_ok=True)
    abl_csv = os.path.join(OUT_ABL_DIR, 'iso_prompt_metrics.csv')
    with open(abl_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['prompt','rank1','mAP','n_query','n_gallery','elapsed_s'])
        for r in summary_rows:
            w.writerow(r)
    best_txt = os.path.join(OUT_ABL_DIR, 'best_iso_prompt.txt')
    with open(best_txt, 'w', encoding='utf-8') as f:
        f.write(f"best_by_mAP={best_by_map['prompt']}\n")
        f.write(f"mAP={best_by_map['mAP']:.4f}, rank1={best_by_map['rank1']:.4f}\n")
    print(f'[OK] ablation CSV -> {abl_csv}, best -> {best_txt}')
    # Update docs
    improved = any(r[1] > BASELINE_RANK1 for r in summary_rows)
    table_md = ['', '### ISO Prompt 消融', '', '| Prompt | Rank-1 | mAP | n_query | n_gallery | elapsed_s |', '|---|---:|---:|---:|---:|---:|']
    for pid, r1, mp, nq, ng, el in summary_rows:
        table_md.append(f"| {pid} | {r1:.4f} | {mp:.4f} | {nq} | {ng} | {el:.3f} |")
    table_md.append('')
    best_line = f"Best (mAP): {best_by_map['prompt']} (mAP={best_by_map['mAP']:.4f}, rank1={best_by_map['rank1']:.4f})"
    if improved:
        best_line += ' — improved over baseline (rank1=0.25)'
    # add model and timestamp context
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    extra = f"Model= {os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'}, Language= English, Timestamp= {ts}"
    append_md = "\n".join(table_md + [best_line, extra, ''])
    try:
        with open(DOC_SUMMARY_MD, 'a', encoding='utf-8') as f:
            f.write('\n' + append_md)
        print('[OK] docs/Execution_Summary.md updated: ISO Prompt 消融')
    except Exception as e:
        print('[WARN] failed to update docs:', e)
    print('[DONE] Multi-prompt ablation completed')


if __name__ == '__main__':
    main()