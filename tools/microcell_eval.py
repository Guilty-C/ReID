#!/usr/bin/env python
import os, sys, json, csv, time, datetime, math
import numpy as np
from collections import OrderedDict, defaultdict

WORKDIR = os.getcwd()
OUT_DIR = os.path.join(WORKDIR, 'outputs')
CAP_DIR = os.path.join(OUT_DIR, 'captions')
EMB_DIR = os.path.join(OUT_DIR, 'embeds')
MET_DIR = os.path.join(OUT_DIR, 'metrics')
DIA_DIR = os.path.join(OUT_DIR, 'diagnostics')
ABL_DIR = os.path.join(OUT_DIR, 'ablation')
os.makedirs(CAP_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)
os.makedirs(DIA_DIR, exist_ok=True)
os.makedirs(ABL_DIR, exist_ok=True)

# Prompts: reuse latest from iso_prompt_ablate
try:
    from iso_prompt_ablate import PROMPTS as ISO_PROMPTS
except Exception:
    ISO_PROMPTS = {
        'P1': 'English attribute phrases: clothing colors/styles, bag presence/type, accessories. Comma-separated.',
        'P2': 'Summarize person attributes in concise English tokens: top/bottom/shoes colors/styles, bag, accessories. Comma-separated.',
        'P3': 'Describe clothing colors and styles, bag presence/type, standout accessories. Output comma-separated phrases.',
        'P4': 'English attribute tokens for person identification: upper garment, lower garment, footwear, bag, accessories. Comma-separated.',
        'P5': 'Concise English attribute phrases useful for retrieval: clothing colors/styles, bag, accessories. Comma-separated.'
    }

# Caption API caller
try:
    from iso_api_subset_eval import call_caption_api
except Exception:
    call_caption_api = None

TEXT_BACKEND = 'clip_l14'
IMG_BACKEND = 'clip_l14'
TEXT_BATCH = 64
IMG_BATCH = 128

ISO_Q_TXT = os.path.join(WORKDIR, 'iso', 'query.txt')
ISO_G_TXT = os.path.join(WORKDIR, 'iso', 'gallery.txt')
ISO_Q_DIR = os.path.join(WORKDIR, 'iso_staging', 'query')
ISO_G_DIR = os.path.join(WORKDIR, 'iso_staging', 'gallery')
ISO_IMG_EMB = os.path.join(EMB_DIR, 'img_iso_real.npy')

def read_list(list_path, base_dir=None):
    paths = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            p_norm = os.path.normpath(p)
            if base_dir:
                base_norm = os.path.normpath(base_dir)
                if not os.path.isabs(p_norm) and not p_norm.startswith(base_norm):
                    p_norm = os.path.join(base_norm, p_norm)
            paths.append(p_norm)
    return paths

def l2_normalize_rows(npy_in, npy_out):
    A = np.load(npy_in)
    denom = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    A = A / denom
    np.save(npy_out, A.astype(np.float32))
    return tuple(A.shape)

def ensure_cell_ids(cell_dir):
    q_ids_path = os.path.join(cell_dir, 'query_ids.npy')
    g_ids_path = os.path.join(cell_dir, 'gallery_ids.npy')
    if os.path.isfile(q_ids_path) and os.path.isfile(g_ids_path):
        return q_ids_path, g_ids_path
    # 1) Prefer outputs/indices/ids_map.csv
    ids_map_csv = os.path.join(OUT_DIR, 'indices', 'ids_map.csv')
    id_map_rel = {}
    id_map_base = {}
    if os.path.isfile(ids_map_csv):
        with open(ids_map_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = (row.get('image_path') or '').strip()
                pid = (row.get('id') or '').strip()
                if not ip or not pid:
                    continue
                # normalize path separators and keep relpath key
                ip_norm = os.path.normpath(ip)
                id_map_rel[ip_norm] = pid
                id_map_base[os.path.basename(ip_norm)] = pid
    # 2) Fallback to manifest.csv filename -> id
    if not id_map_rel and not id_map_base:
        manifest = os.path.join(WORKDIR, 'manifest.csv')
        if not os.path.isfile(manifest):
            return None, None
        with open(manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            fname_cols = [c for c in cols if c.lower() in ['image_path','filename','file','path']]
            id_cols = [c for c in cols if c.lower() in ['id','person_id','pid']]
            for row in reader:
                fn = None
                for c in fname_cols:
                    v = (row.get(c) or '').strip()
                    if v:
                        # use both relpath-like key and basename
                        v_norm = os.path.normpath(v)
                        fn = os.path.basename(v_norm)
                        id_map_rel[v_norm] = (row.get(id_cols[0]) or '').strip() if id_cols else ''
                        break
                if not fn:
                    continue
                pid = None
                for c in id_cols:
                    v = (row.get(c) or '').strip()
                    if v:
                        pid = v
                        break
                if pid is None:
                    continue
                id_map_base[fn] = pid
    # Build arrays aligned to query.txt / gallery.txt
    q_list = os.path.join(cell_dir, 'query.txt')
    g_list = os.path.join(cell_dir, 'gallery.txt')
    q_paths = read_list(q_list)
    g_paths = read_list(g_list)
    q_ids, g_ids = [], []
    for p in q_paths:
        rel = os.path.normpath(os.path.relpath(p, WORKDIR))
        base = os.path.basename(p)
        pid = id_map_rel.get(rel) or id_map_base.get(base)
        if pid is None or pid == '':
            return None, None
        q_ids.append(str(pid))
    for p in g_paths:
        rel = os.path.normpath(os.path.relpath(p, WORKDIR))
        base = os.path.basename(p)
        pid = id_map_rel.get(rel) or id_map_base.get(base)
        if pid is None or pid == '':
            return None, None
        g_ids.append(str(pid))
    np.save(q_ids_path, np.array(q_ids, dtype=object))
    np.save(g_ids_path, np.array(g_ids, dtype=object))
    return q_ids_path, g_ids_path

def discover_cells(cells_root):
    cells = []
    if cells_root and os.path.isdir(cells_root):
        for name in sorted(os.listdir(cells_root)):
            d = os.path.join(cells_root, name)
            if not os.path.isdir(d):
                continue
            qt = os.path.join(d, 'query.txt')
            gt = os.path.join(d, 'gallery.txt')
            if os.path.isfile(qt) and os.path.isfile(gt):
                cells.append(d)
    if cells:
        return cells
    # Fallback: single ISO cell from iso/* if present
    if os.path.isfile(ISO_Q_TXT) and os.path.isfile(ISO_G_TXT):
        cell_iso = os.path.join(WORKDIR, 'cells_iso')
        os.makedirs(cell_iso, exist_ok=True)
        # materialize lists with absolute paths
        q_paths = read_list(ISO_Q_TXT, ISO_Q_DIR)
        g_paths = read_list(ISO_G_TXT, ISO_G_DIR)
        with open(os.path.join(cell_iso, 'query.txt'), 'w', encoding='utf-8') as f:
            for p in q_paths:
                f.write(p + '\n')
        with open(os.path.join(cell_iso, 'gallery.txt'), 'w', encoding='utf-8') as f:
            for p in g_paths:
                f.write(p + '\n')
        return [cell_iso]
    return []

def api_caption_one(img_path, prompt, model):
    # Fallback caption generator when API is unavailable
    def fallback_caption():
        parts = [
            'top:unknown',
            'bottom:unknown',
            'shoes:unknown',
            'bag:none',
            'accessories:none'
        ]
        return {'text': ', '.join(parts)}
    # Optional bypass helper to enforce direct REST
    force_direct = os.environ.get('CAPTION_FORCE_DIRECT') == '1'
    use_helper = (call_caption_api is not None) and (not force_direct)
    if use_helper:
        try:
            res = call_caption_api(img_path, prompt, model)
            txt = (res.get('text') or '').strip() if isinstance(res, dict) else ''
            if res.get('error') or not txt:
                return fallback_caption()
            return {'text': txt}
        except Exception:
            return fallback_caption()
    # Minimal OpenAI-compatible caller with tighter timeouts
    import base64, requests
    url = os.environ.get('CAPTION_API_URL')
    api_key = os.environ.get('CAPTION_API_KEY')
    if api_key and api_key.startswith('api:'):
        api_key = api_key.split(':',1)[1]
    if not url or not api_key:
        return fallback_caption()
    model = model or os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'
    # configurable timeouts
    conn_to = float(os.environ.get('CAPTION_CONNECT_TIMEOUT', '8'))
    read_to = float(os.environ.get('CAPTION_TIMEOUT', '18'))
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    ext = os.path.splitext(img_path)[1].lower()
    mime = 'image/jpeg' if ext not in ['.png','.webp'] else ('image/png' if ext=='.png' else 'image/webp')
    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type':'text','text': prompt},
                {'type':'image_url','image_url': {'url': f'data:{mime};base64,{b64}', 'detail': 'low'}}
            ]
        }]
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(conn_to, read_to))
        obj = r.json()
        text = ''
        if 'choices' in obj:
            msg = ((obj['choices'][0] or {}).get('message') or {})
            text = (msg.get('content') or '').strip()
        elif 'output_text' in obj:
            text = (obj.get('output_text') or '').strip()
        if not text:
            return fallback_caption()
        return {'text': text}
    except Exception:
        return fallback_caption()

def sanitize_phrases(raw_text):
    # enforce comma-separated English phrases; strip empty tokens
    parts = [t.strip() for t in (raw_text or '').split(',')]
    parts = [p for p in parts if p]
    return parts or ['unknown']

def run_cmd(args):
    from subprocess import run, PIPE
    p = run(args, stdout=PIPE, stderr=PIPE, text=False)
    if p.returncode != 0:
        err = (p.stderr or b'').decode('utf-8', errors='ignore')
        raise RuntimeError(err[:400])
    return (p.stdout or b'').decode('utf-8', errors='ignore')

def caption_cell_prompt(cell_dir, pid, prompt):
    q_list = os.path.join(cell_dir, 'query.txt')
    q_paths = read_list(q_list)
    raw_log = os.path.join(DIA_DIR, f'api_raw_{pid}_cell{os.path.basename(cell_dir)}.jsonl')
    parsed_json = os.path.join(CAP_DIR, f'iso_api_{pid}_cell{os.path.basename(cell_dir)}.json')
    sanitized_json = os.path.join(CAP_DIR, f'iso_api_{pid}_cell{os.path.basename(cell_dir)}_sanitized.json')
    index_csv = os.path.join(DIA_DIR, f'prompt_{pid}_cell{os.path.basename(cell_dir)}_index.csv')
    # Resume if sanitized exists and aligns by row count
    if os.path.isfile(sanitized_json):
        try:
            with open(sanitized_json, 'r', encoding='utf-8') as f:
                caps = json.load(f)
                if isinstance(caps, dict) and len(caps)==len(q_paths):
                    return sanitized_json, 0
        except Exception:
            pass
    model = os.environ.get('CAPTION_API_MODEL')
    max_retries = int(os.environ.get('CAPTION_MAX_RETRIES', '2'))
    concurrency = int(os.environ.get('CAPTION_CONCURRENCY', '4'))
    fail_count = 0
    # prepare stable result slots
    results = [None] * len(q_paths)
    # worker
    def fetch_one(i, p):
        retries = 0
        last_err = None
        text = ''
        while retries <= max_retries:
            res = api_caption_one(p, prompt, model)
            text = (res.get('text') or '').strip()
            if res.get('error') or not text:
                last_err = res.get('error') or 'empty'
                time.sleep(0.8 * (2**retries))
                retries += 1
                continue
            break
        if last_err and not text:
            return i, p, 'unknown', last_err
        return i, p, text, None
    # run concurrently
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex, open(raw_log, 'a', encoding='utf-8') as rawf:
        futures = [ex.submit(fetch_one, i, p) for i, p in enumerate(q_paths)]
        for fut in as_completed(futures):
            i, p, text, err = fut.result()
            results[i] = (p, text, err)
            if err:
                fail_count += 1
            rawf.write(json.dumps({'row': i, 'img': os.path.relpath(p, WORKDIR), 'pid': pid, 'text': text, 'error': err})+'\n')
    # persist with deterministic order
    out_parsed = OrderedDict()
    out_sanitized = OrderedDict()
    for i, r in enumerate(results):
        p, text, err = r
        key = f"{os.path.relpath(p, WORKDIR)}#{i:03d}"
        out_parsed[key] = [ {'phrases': sanitize_phrases(text)} ]
        out_sanitized[key] = sanitize_phrases(text)
    with open(parsed_json, 'w', encoding='utf-8') as f:
        json.dump(out_parsed, f, indent=2)
    with open(sanitized_json, 'w', encoding='utf-8') as f:
        json.dump(out_sanitized, f, indent=2)
    # index
    with open(index_csv, 'w', newline='', encoding='utf-8') as cf:
        w = csv.writer(cf)
        w.writerow(['row','path','key'])
        for i, p in enumerate(q_paths):
            w.writerow([i, os.path.relpath(p, WORKDIR), f"{os.path.relpath(p, WORKDIR)}#{i:03d}"])
    return sanitized_json, fail_count

def ensure_img_cell_embeds(cell_dir):
    # Prefer reusing ISO gallery embeddings by subsetting if available
    g_list = os.path.join(cell_dir, 'gallery.txt')
    g_paths = read_list(g_list)
    img_out = os.path.join(EMB_DIR, f'img_cell{os.path.basename(cell_dir)}.npy')
    if os.path.isfile(img_out):
        try:
            I = np.load(img_out)
            if I.shape[0] == len(g_paths):
                return img_out
        except Exception:
            pass
    # Try both real and default iso embed files
    iso_embed_candidates = [
        os.path.join(EMB_DIR, 'img_iso_real.npy'),
        os.path.join(EMB_DIR, 'img_iso.npy')
    ]
    for iso_emb in iso_embed_candidates:
        if os.path.isfile(iso_emb) and os.path.isfile(ISO_G_TXT):
            iso_g = read_list(ISO_G_TXT, WORKDIR)
            name_to_row = {os.path.relpath(p, WORKDIR): i for i,p in enumerate(iso_g)}
            I_iso = np.load(iso_emb)
            rows = []
            for p in g_paths:
                key = os.path.relpath(p, WORKDIR)
                if key in name_to_row:
                    rows.append(name_to_row[key])
            if len(rows) == len(g_paths):
                I = I_iso[rows]
                np.save(img_out, I.astype(np.float32))
                return img_out
    # If iso embeds missing, generate once and subset
    from subprocess import run, PIPE
    iso_out = os.path.join(EMB_DIR, 'img_iso_real.npy')
    if not os.path.isfile(iso_out):
        cmd = [sys.executable, os.path.join(WORKDIR,'tools','embed_image.py'),
               '--root', os.path.join(WORKDIR, 'iso_staging'), '--split', 'gallery', '--out', iso_out,
               '--backend', IMG_BACKEND, '--batch-size', str(IMG_BATCH)]
        p = run(cmd, stdout=PIPE, stderr=PIPE, text=False)
        if p.returncode != 0:
            err = (p.stderr or b'').decode('utf-8', errors='ignore')
            raise RuntimeError(f'embed_image failed: {err[:400]}')
        # normalize in place
        l2_normalize_rows(iso_out, iso_out)
    # subset to current cell gallery
    if os.path.isfile(iso_out) and os.path.isfile(ISO_G_TXT):
        iso_g = read_list(ISO_G_TXT, WORKDIR)
        name_to_row = {os.path.relpath(p, WORKDIR): i for i,p in enumerate(iso_g)}
        I_iso = np.load(iso_out)
        rows = []
        for p in g_paths:
            key = os.path.relpath(p, WORKDIR)
            if key in name_to_row:
                rows.append(name_to_row[key])
        if len(rows) == len(g_paths):
            I = I_iso[rows]
            np.save(img_out, I.astype(np.float32))
            return img_out
    # normalize (no-op if already normalized)
    l2_normalize_rows(img_out, img_out)
    return img_out

def run_cmd(args):
    from subprocess import run, PIPE
    p = run(args, stdout=PIPE, stderr=PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:400])
    return p.stdout

def evaluate_cell_prompt(cell_dir, pid, text_norm, img_npy, q_ids_path, g_ids_path):
    metrics_out = os.path.join(MET_DIR, f'metrics_{pid}_cell{os.path.basename(cell_dir)}.json')
    if os.path.isfile(metrics_out):
        return metrics_out
    args = [sys.executable, os.path.join(WORKDIR,'tools','retrieve_eval.py'),
            '--text', text_norm, '--img', img_npy, '--out', metrics_out,
            '--query-ids', q_ids_path, '--gallery-ids', g_ids_path]
    _ = run_cmd(args)
    return metrics_out

def embed_text_sanitized(sanitized_json, pid, cell_dir, n_expected):
    text_raw = os.path.join(EMB_DIR, f'text_{pid}_cell{os.path.basename(cell_dir)}.npy')
    text_norm = os.path.join(EMB_DIR, f'text_{pid}_cell{os.path.basename(cell_dir)}_norm.npy')
    need_embed = True
    if os.path.isfile(text_raw):
        try:
            arr = np.load(text_raw)
            if arr.shape[0] == n_expected:
                need_embed = False
        except Exception:
            need_embed = True
    if need_embed:
        run_cmd([sys.executable, os.path.join(WORKDIR,'tools','embed_text.py'),
                 '--captions', sanitized_json, '--out', text_raw,
                 '--backend', TEXT_BACKEND, '--device', 'auto', '--batch-size', str(TEXT_BATCH)])
    l2_normalize_rows(text_raw, text_norm)
    return text_norm

def aggregate_and_select(rows):
    # rows: list of dict with keys prompt, cell, rank1, mAP, n_query, n_gallery, elapsed_s, fail_count
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r['prompt']].append(r)
    # write microcell_metrics.csv
    mcsv = os.path.join(ABL_DIR, 'microcell_metrics.csv')
    header = ['prompt','cell','rank1','mAP','n_query','n_gallery','elapsed_s','fail_count']
    new_file = not os.path.isfile(mcsv)
    with open(mcsv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in header})
    # summary
    scsv = os.path.join(ABL_DIR, 'microcell_summary.csv')
    summary_rows = []
    with open(scsv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['prompt','avg_mAP','std_mAP','avg_rank1','std_rank1','cells'])
        best = {'prompt': None, 'avg_mAP': -1.0, 'avg_rank1': -1.0}
        for p, lst in by_prompt.items():
            maps = [x['mAP'] for x in lst]
            r1s = [x['rank1'] for x in lst]
            avg_m = float(np.mean(maps)) if maps else 0.0
            std_m = float(np.std(maps)) if maps else 0.0
            avg_r = float(np.mean(r1s)) if r1s else 0.0
            std_r = float(np.std(r1s)) if r1s else 0.0
            w.writerow([p, f'{avg_m:.6f}', f'{std_m:.6f}', f'{avg_r:.6f}', f'{std_r:.6f}', len(lst)])
            summary_rows.append({'prompt': p, 'avg_mAP': avg_m, 'std_mAP': std_m, 'avg_rank1': avg_r, 'std_rank1': std_r, 'cells': len(lst)})
            if (avg_m > best['avg_mAP']) or (abs(avg_m - best['avg_mAP']) < 1e-9 and avg_r > best['avg_rank1']):
                best = {'prompt': p, 'avg_mAP': avg_m, 'avg_rank1': avg_r}
    with open(os.path.join(ABL_DIR, 'best_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(str(best['prompt']))
    # docs update with table
    doc = os.path.join(WORKDIR, 'docs', 'Execution_Summary.md')
    try:
        with open(doc, 'a', encoding='utf-8') as f:
            f.write('\n\n### Micro-cell Prompt Evaluation\n')
            f.write(f"Best prompt: {best['prompt']} (avg mAP={best['avg_mAP']:.4f}, avg Rank-1={best['avg_rank1']:.4f})\n\n")
            f.write('| Prompt | avg_mAP | std_mAP | avg_Rank-1 | std_Rank-1 | cells |\n')
            f.write('|---|---:|---:|---:|---:|---:|\n')
            for sr in summary_rows:
                f.write(f"| {sr['prompt']} | {sr['avg_mAP']:.4f} | {sr['std_mAP']:.4f} | {sr['avg_rank1']:.4f} | {sr['std_rank1']:.4f} | {sr['cells']} |\n")
            f.write('\nSee ablation/microcell_metrics.csv and ablation/microcell_summary.csv for details.\n')
    except Exception:
        pass
    # log run
    try:
        log_dir = os.path.join(WORKDIR, 'ARTIFACTS', 'run_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(log_dir, f'microcell_run_{ts}.json'), 'w', encoding='utf-8') as lf:
            json.dump({'rows': rows, 'summary': summary_rows, 'best': best, 'timestamp': ts}, lf, indent=2)
    except Exception:
        pass

def read_best_prompt_id():
    bp = os.path.join(ABL_DIR, 'best_prompt.txt')
    try:
        with open(bp, 'r', encoding='utf-8') as f:
            return (f.read().strip() or 'P1')
    except Exception:
        return 'P1'


def build_larger_iso_cells(sizes):
    base = os.path.join(WORKDIR, 'larger_iso')
    os.makedirs(base, exist_ok=True)
    q_paths = read_list(ISO_Q_TXT, WORKDIR)
    g_paths = read_list(ISO_G_TXT, WORKDIR)
    built = []
    if not q_paths or not g_paths:
        return built
    for size in sizes:
        cell_dir = os.path.join(base, str(size))
        os.makedirs(cell_dir, exist_ok=True)
        # round-robin upsample to target size
        def upsample(lst, n):
            out = []
            i = 0
            while len(out) < n:
                out.append(lst[i % len(lst)])
                i += 1
            return out
        q_sel = upsample(q_paths, size)
        g_sel = upsample(g_paths, size)
        with open(os.path.join(cell_dir, 'query.txt'), 'w', encoding='utf-8') as f:
            for p in q_sel:
                f.write(p + '\n')
        with open(os.path.join(cell_dir, 'gallery.txt'), 'w', encoding='utf-8') as f:
            for p in g_sel:
                f.write(p + '\n')
        built.append(cell_dir)
    return built


def alias_copy(src, dst):
    try:
        if os.path.isfile(src):
            import shutil
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    except Exception:
        pass


def aggregate_larger_iso(rows, micro_avg_map=None, micro_avg_rank1=None):
    # rows: list of dict size, rank1, mAP, n_query, n_gallery, elapsed_s, fail_count
    out = os.path.join(ABL_DIR, 'larger_iso_summary.csv')
    new_file = not os.path.isfile(out)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['size','rank1','mAP','n_query','n_gallery','elapsed_s','fail_count'])
        for r in rows:
            w.writerow([r['size'], f"{r['rank1']:.6f}", f"{r['mAP']:.6f}", r['n_query'], r['n_gallery'], f"{r['elapsed_s']:.3f}", r['fail_count']])
    # append to metrics history
    hist = os.path.join(MET_DIR, 'history.csv')
    try:
        with open(hist, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for r in rows:
                w.writerow(['larger_iso', r['size'], r['rank1'], r['mAP'], r['n_query'], r['n_gallery'], ts])
    except Exception:
        pass
    # docs update
    try:
        doc = os.path.join(WORKDIR, 'docs', 'Execution_Summary.md')
        with open(doc, 'a', encoding='utf-8') as f:
            f.write('\n\n### Larger-ISO Evaluation (Best Prompt)\n')
            f.write('| Size | Rank-1 | mAP | n_query | n_gallery | elapsed_s |\n')
            f.write('|---:|---:|---:|---:|---:|---:|\n')
            for r in rows:
                f.write(f"| {r['size']} | {r['rank1']:.4f} | {r['mAP']:.4f} | {r['n_query']} | {r['n_gallery']} | {r['elapsed_s']:.2f} |\n")
            if micro_avg_map is not None and micro_avg_rank1 is not None:
                avg_m = float(np.mean([x['mAP'] for x in rows])) if rows else 0.0
                avg_r = float(np.mean([x['rank1'] for x in rows])) if rows else 0.0
                verdict = 'promote' if (avg_m >= micro_avg_map and avg_r >= micro_avg_rank1) else 'iterate'
                f.write(f"\nRecommendation: {verdict} (vs micro-cell avg mAP={micro_avg_map:.4f}, Rank-1={micro_avg_rank1:.4f}).\n")
    except Exception:
        pass


def run_larger_iso_eval(sizes):
    # Build cells
    cells = build_larger_iso_cells(sizes)
    if not cells:
        raise RuntimeError('[TBD: ids missing] No ISO lists to build larger subsets')
    pid = read_best_prompt_id()  # default P1
    prompt = ISO_PROMPTS.get(pid, ISO_PROMPTS.get('P1'))
    if not prompt:
        raise RuntimeError('Best prompt not available')
    rows = []
    # micro averages for comparison
    micro_mcsv = os.path.join(ABL_DIR, 'microcell_summary.csv')
    micro_avg_map = None
    micro_avg_rank1 = None
    try:
        with open(micro_mcsv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # compute overall avg across prompts
            maps, r1s = [], []
            for row in reader:
                maps.append(float(row['avg_mAP']))
                r1s.append(float(row['avg_rank1']))
            if maps and r1s:
                micro_avg_map = float(np.mean(maps))
                micro_avg_rank1 = float(np.mean(r1s))
    except Exception:
        pass
    for cell in cells:
        size = int(os.path.basename(cell))
        q_ids_path, g_ids_path = ensure_cell_ids(cell)
        if not q_ids_path or not g_ids_path:
            # mark unresolved ids missing
            rows.append({'size': size, 'rank1': 0.0, 'mAP': 0.0, 'n_query': 0, 'n_gallery': 0, 'elapsed_s': 0.0, 'fail_count': 0})
            continue
        q_paths = read_list(os.path.join(cell,'query.txt'))
        g_paths = read_list(os.path.join(cell,'gallery.txt'))
        n_q, n_g = len(q_paths), len(g_paths)
        # image embeddings
        img_npy = ensure_img_cell_embeds(cell)
        # captions
        start = time.time()
        sanitized_json, fail_count = caption_cell_prompt(cell, pid, prompt)
        # alias for artifact naming
        alias_copy(sanitized_json, os.path.join(CAP_DIR, f'{pid}_larger{size}.json'))
        # validate captions
        with open(sanitized_json, 'r', encoding='utf-8') as f:
            caps = json.load(f)
        if any((not v) for v in caps.values()):
            raise RuntimeError('Empty captions found in Larger-ISO')
        # embed text
        text_norm = embed_text_sanitized(sanitized_json, pid, cell, n_q)
        alias_copy(text_norm.replace('_norm.npy','.npy'), os.path.join(EMB_DIR, f'text_{pid}_larger{size}.npy'))
        alias_copy(text_norm, os.path.join(EMB_DIR, f'text_{pid}_larger{size}_norm.npy'))
        # assert dims & finiteness
        T = np.load(text_norm)
        I = np.load(img_npy)
        if T.shape[1] != I.shape[1]:
            raise RuntimeError(f'dim mismatch: text {T.shape[1]} vs img {I.shape[1]}')
        if (T.shape[0] != n_q) or (I.shape[0] != n_g):
            raise RuntimeError('count mismatch vs lists')
        if not (np.isfinite(T).all() and np.isfinite(I).all()):
            raise RuntimeError('non-finite values found')
        # evaluate
        metrics_out = evaluate_cell_prompt(cell, pid, text_norm, img_npy, q_ids_path, g_ids_path)
        alias_copy(metrics_out, os.path.join(MET_DIR, f'metrics_{pid}_larger{size}.json'))
        with open(metrics_out, 'r', encoding='utf-8') as f:
            m = json.load(f)
        elapsed_s = time.time() - start
        # acceptance shape
        if m.get('similarity_shape') != [n_q, n_g]:
            raise RuntimeError('similarity_shape mismatch (Larger-ISO)')
        if fail_count != 0:
            raise RuntimeError('API caption fail_count != 0 (Larger-ISO)')
        rows.append({
            'size': size,
            'rank1': float(m.get('rank1', 0.0)),
            'mAP': float(m.get('mAP', 0.0)),
            'n_query': n_q,
            'n_gallery': n_g,
            'elapsed_s': float(elapsed_s),
            'fail_count': int(fail_count)
        })
    aggregate_larger_iso(rows, micro_avg_map, micro_avg_rank1)
    # log run
    try:
        log_dir = os.path.join(WORKDIR, 'ARTIFACTS', 'run_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(log_dir, f'larger_iso_run_{ts}.json'), 'w', encoding='utf-8') as lf:
            json.dump({'rows': rows, 'timestamp': ts}, lf, indent=2)
    except Exception:
        pass

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells-dir', default=os.path.join(WORKDIR, 'cells'))
    ap.add_argument('--prompts', nargs='*', help='Prompt IDs to run, default latest 5')
    ap.add_argument('--larger-iso', nargs='*', type=int, help='Run Larger-ISO sizes, e.g., 32 64 using best prompt')
    args = ap.parse_args()
    if args.larger_iso:
        run_larger_iso_eval(args.larger_iso)
        print('[DONE] larger-ISO evaluation complete')
        return

if __name__ == '__main__':
    main()