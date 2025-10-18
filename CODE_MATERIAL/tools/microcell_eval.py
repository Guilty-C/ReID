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
            if base_dir and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            paths.append(p)
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
    if call_caption_api:
        return call_caption_api(img_path, prompt, model)
    # Minimal OpenAI-compatible caller
    import base64, requests
    url = os.environ.get('CAPTION_API_URL')
    api_key = os.environ.get('CAPTION_API_KEY')
    if api_key and api_key.startswith('api:'):
        api_key = api_key.split(':',1)[1]
    if not url or not api_key:
        return {'error': 'MISSING_ENV', 'text': '[API_ERROR] missing env'}
    model = model or os.environ.get('CAPTION_API_MODEL') or 'gpt-4o-mini'
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
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        obj = r.json()
        # naive parse
        text = ''
        if 'choices' in obj:
            msg = ((obj['choices'][0] or {}).get('message') or {})
            text = (msg.get('content') or '').strip()
        elif 'output_text' in obj:
            text = (obj.get('output_text') or '').strip()
        return {'text': text or 'unknown', 'http': getattr(r, 'status_code', None)}
    except Exception as e:
        return {'error': str(e), 'text': 'unknown'}

def sanitize_phrases(raw_text):
    # enforce comma-separated English phrases; strip empty tokens
    parts = [t.strip() for t in (raw_text or '').split(',')]
    parts = [p for p in parts if p]
    return parts or ['unknown']

def caption_cell_prompt(cell_dir, pid, prompt):
    q_list = os.path.join(cell_dir, 'query.txt')
    q_paths = read_list(q_list)
    raw_log = os.path.join(DIA_DIR, f'api_raw_{pid}_cell{os.path.basename(cell_dir)}.jsonl')
    parsed_json = os.path.join(CAP_DIR, f'iso_api_{pid}_cell{os.path.basename(cell_dir)}.json')
    sanitized_json = os.path.join(CAP_DIR, f'iso_api_{pid}_cell{os.path.basename(cell_dir)}_sanitized.json')
    index_csv = os.path.join(DIA_DIR, f'prompt_{pid}_cell{os.path.basename(cell_dir)}_index.csv')
    # Resume if sanitized exists and aligns
    if os.path.isfile(sanitized_json):
        try:
            with open(sanitized_json, 'r', encoding='utf-8') as f:
                caps = json.load(f)
                if isinstance(caps, dict) and len(caps)==len(q_paths):
                    return sanitized_json, 0
        except Exception:
            pass
    model = os.environ.get('CAPTION_API_MODEL')
    fail_count = 0
    out_parsed = OrderedDict()
    out_sanitized = OrderedDict()
    t0 = time.time()
    # progress bar if tty
    use_bar = sys.stdout.isatty() or os.environ.get('TQDM_FORCE')=='1'
    bar = None
    if use_bar:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(q_paths), desc=f'Caption {pid}/{os.path.basename(cell_dir)}', unit='img', leave=False)
        except Exception:
            bar = None
    with open(raw_log, 'a', encoding='utf-8') as rawf:
        for p in q_paths:
            retries = 0
            last_err = None
            while retries < 3:
                res = api_caption_one(p, prompt, model)
                text = (res.get('text') or '').strip()
                if res.get('error') or not text:
                    last_err = res.get('error') or 'empty'
                    time.sleep(1.5 * (2**retries))
                    retries += 1
                    continue
                # ok
                break
            if last_err and not text:
                fail_count += 1
                text = 'unknown'
            rawf.write(json.dumps({'img': os.path.relpath(p, WORKDIR), 'pid': pid, 'text': text, 'error': res.get('error')})+'\n')
            out_parsed[os.path.relpath(p, WORKDIR)] = [ {'phrases': sanitize_phrases(text)} ]
            out_sanitized[os.path.relpath(p, WORKDIR)] = sanitize_phrases(text)
            if bar: bar.update(1)
    # persist
    with open(parsed_json, 'w', encoding='utf-8') as f:
        json.dump(out_parsed, f, indent=2)
    with open(sanitized_json, 'w', encoding='utf-8') as f:
        json.dump(out_sanitized, f, indent=2)
    # index
    with open(index_csv, 'w', newline='', encoding='utf-8') as cf:
        w = csv.writer(cf)
        w.writerow(['row','path'])
        for i, p in enumerate(q_paths):
            w.writerow([i, os.path.relpath(p, WORKDIR)])
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
    if os.path.isfile(ISO_IMG_EMB) and os.path.isfile(ISO_G_TXT):
        iso_g = read_list(ISO_G_TXT, ISO_G_DIR)
        name_to_row = {os.path.relpath(p, ISO_G_DIR): i for i,p in enumerate(iso_g)}
        I_iso = np.load(ISO_IMG_EMB)
        rows = []
        for p in g_paths:
            key = os.path.relpath(p, ISO_G_DIR)
            if key in name_to_row:
                rows.append(name_to_row[key])
        if len(rows) == len(g_paths):
            I = I_iso[rows]
            np.save(img_out, I.astype(np.float32))
            return img_out
    # Fallback: call embed_image.py on cell gallery root
    from subprocess import run, PIPE
    elapsed_start = time.time()
    cmd = [sys.executable, os.path.join(WORKDIR,'tools','embed_image.py'),
           '--root', os.path.dirname(g_paths[0]), '--split', 'gallery', '--out', img_out,
           '--backend', IMG_BACKEND, '--batch-size', str(IMG_BATCH)]
    p = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'embed_image failed: {p.stderr[:400]}')
    # normalize
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
            if (avg_m > best['avg_mAP']) or (abs(avg_m - best['avg_mAP']) < 1e-9 and avg_r > best['avg_rank1']):
                best = {'prompt': p, 'avg_mAP': avg_m, 'avg_rank1': avg_r}
    with open(os.path.join(ABL_DIR, 'best_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(str(best['prompt']))
    # docs update
    doc = os.path.join(WORKDIR, 'docs', 'Execution_Summary.md')
    try:
        with open(doc, 'a', encoding='utf-8') as f:
            f.write('\n\n### Micro-cell Prompt Evaluation\n')
            f.write(f"Best prompt: {best['prompt']} (avg mAP={best['avg_mAP']:.4f}, avg Rank-1={best['avg_rank1']:.4f})\n\n")
            f.write('See ablation/microcell_metrics.csv and ablation/microcell_summary.csv for details.\n')
    except Exception:
        pass

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells-dir', default=os.path.join(WORKDIR, 'cells'))
    ap.add_argument('--prompts', nargs='*', help='Prompt IDs to run, default latest 5')
    args = ap.parse_args()
    cells = discover_cells(args.cells_dir)
    if not cells:
        print('[WARN] No cells found; using fallback ISO cell if available.')
        cells = discover_cells(None)
    if not cells:
        raise RuntimeError('No cells available to run')
    # Prepare prompts
    pids = args.prompts or list(ISO_PROMPTS.keys())[:5]
    prompts = {pid: ISO_PROMPTS[pid] for pid in pids if pid in ISO_PROMPTS}
    if not prompts:
        raise RuntimeError('No valid prompts to run')
    rows = []
    for cell in cells:
        print(f'\n[Cell] {cell}')
        q_ids_path, g_ids_path = ensure_cell_ids(cell)
        if not q_ids_path or not g_ids_path:
            print(f'[SKIP] IDs missing for {cell} -> [TBD: ids missing]')
            continue
        q_paths = read_list(os.path.join(cell,'query.txt'))
        g_paths = read_list(os.path.join(cell,'gallery.txt'))
        n_q, n_g = len(q_paths), len(g_paths)
        print(f'[INFO] n_query={n_q}, n_gallery={n_g}')
        img_npy = ensure_img_cell_embeds(cell)
        for pid, prompt in prompts.items():
            print(f'[RUN] {pid} on {os.path.basename(cell)}')
            start = time.time()
            sanitized_json, fail_count = caption_cell_prompt(cell, pid, prompt)
            # Assert no empty captions
            with open(sanitized_json, 'r', encoding='utf-8') as f:
                caps = json.load(f)
            if any((not v) for v in caps.values()):
                raise RuntimeError('Empty captions found')
            text_norm = embed_text_sanitized(sanitized_json, pid, cell, n_q)
            # validate dims & finiteness
            T = np.load(text_norm)
            I = np.load(img_npy)
            if T.shape[1] != I.shape[1]:
                raise RuntimeError(f'dim mismatch: text {T.shape[1]} vs img {I.shape[1]}')
            if (T.shape[0] != n_q) or (I.shape[0] != n_g):
                raise RuntimeError('count mismatch vs lists')
            if not (np.isfinite(T).all() and np.isfinite(I).all()):
                raise RuntimeError('non-finite values found')
            metrics_out = evaluate_cell_prompt(cell, pid, text_norm, img_npy, q_ids_path, g_ids_path)
            with open(metrics_out, 'r', encoding='utf-8') as f:
                m = json.load(f)
            elapsed_s = time.time() - start
            rows.append({
                'prompt': pid,
                'cell': os.path.basename(cell),
                'rank1': float(m.get('rank1', 0.0)),
                'mAP': float(m.get('mAP', 0.0)),
                'n_query': n_q,
                'n_gallery': n_g,
                'elapsed_s': float(elapsed_s),
                'fail_count': int(fail_count)
            })
            # Acceptance checks
            if m.get('similarity_shape') != [n_q, n_g]:
                raise RuntimeError('similarity_shape mismatch')
            if fail_count != 0:
                raise RuntimeError('API caption fail_count != 0')
    if rows:
        aggregate_and_select(rows)
    print('[DONE] micro-cell evaluation complete')

if __name__ == '__main__':
    main()