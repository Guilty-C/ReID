#!/usr/bin/env python
import argparse, os, sys, time, json, subprocess, shlex
from pathlib import Path
import numpy as np

def read_cfg_dataset_root(cfg_path: str) -> str:
    try:
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        root = cfg.get('dataset', {}).get('root', '')
        return root or ''
    except Exception:
        return ''


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd_list, cwd=None):
    start = time.time()
    proc = subprocess.run(cmd_list, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = time.time()
    return proc.returncode, proc.stdout, end - start


def main():
    ap = argparse.ArgumentParser(description='End-to-end experiment: LLM captions → embeds → eval → logs')
    ap.add_argument('--cfg', default='configs/reid.yaml')
    ap.add_argument('--dataset-root', default='')
    ap.add_argument('--subset', choices=['gold','full'], default='gold')
    ap.add_argument('--prompt', default='')
    ap.add_argument('--prompt-file', default='')
    # Captioning options
    ap.add_argument('--caption-mode', choices=['api','clip_attr','desc','salient','json'], default='api')
    ap.add_argument('--api-url', default='')
    ap.add_argument('--api-key', default='')
    ap.add_argument('--api-model', default='')
    ap.add_argument('--clip-model', default='ViT-L/14')
    ap.add_argument('--clip-device', default='auto')
    # Embedding options
    ap.add_argument('--text-backend', default='')
    ap.add_argument('--img-backend', default='')
    ap.add_argument('--outdir', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size-text', type=int, default=64)
    ap.add_argument('--batch-size-image', type=int, default=128)
    ap.add_argument('--subset-count', type=int, default=10)
    # Evaluation options
    ap.add_argument('--re-ranking', dest='re_ranking', action='store_true', help='Enable re-ranking in evaluation')
    ap.add_argument('--no-re-ranking', dest='re_ranking', action='store_false', help='Disable re-ranking (default)')
    ap.add_argument('--r1-thr', type=float, default=0.30, help='Gate threshold for Rank-1')
    ap.add_argument('--map-thr', type=float, default=0.40, help='Gate threshold for mAP')
    ap.set_defaults(re_ranking=False)
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    os.chdir(workdir)

    dataset_root = args.dataset_root or read_cfg_dataset_root(args.cfg)
    if not dataset_root:
        print('Error: dataset.root not provided and not found in config')
        sys.exit(2)

    ts = time.strftime('%Y%m%d-%H%M%S')
    exp_id = f'EXP_{ts}'
    # Always create a dedicated subfolder per experiment under the provided base outdir
    base_outdir = Path(args.outdir or Path('outputs') / 'experiments')
    outdir = base_outdir / exp_id
    ensure_dir(outdir)
    ensure_dir(outdir / 'metrics')
    ensure_dir(outdir / 'embeds')
    ensure_dir(outdir / 'captions')
    ensure_dir(outdir / 'logs')

    # Resolve backends (fallback to config defaults if empty)
    if not args.text_backend or not args.img_backend:
        try:
            import yaml
            with open(args.cfg, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            args.text_backend = args.text_backend or cfg.get('embed', {}).get('text', 'clip_l14')
            args.img_backend = args.img_backend or cfg.get('embed', {}).get('image', 'clip_l14')
        except Exception:
            args.text_backend = args.text_backend or 'clip_l14'
            args.img_backend = args.img_backend or 'clip_l14'

    # Paths
    captions_json = outdir / 'captions' / 'captions_query.json'
    text_embeds = outdir / 'embeds' / 'text_q.npy'
    img_embeds = outdir / 'embeds' / 'img_g.npy'
    metrics_json = outdir / 'metrics' / 'metrics.json'
    sim_npy = outdir / 'embeds' / 'similarity_qg.npy'
    run_log = outdir / 'run.md'
    q_ids_path = outdir / 'embeds' / 'query_ids.npy'
    g_ids_path = outdir / 'embeds' / 'gallery_ids.npy'
    g_paths_txt = outdir / 'embeds' / 'gallery_paths.txt'

    # 1) Caption via API
    # Propagate env for helper to avoid MISSING_ENV
    if args.api_url:
        os.environ['CAPTION_API_URL'] = args.api_url
    if args.api_key:
        os.environ['CAPTION_API_KEY'] = args.api_key
    if args.api_model:
        os.environ['CAPTION_API_MODEL'] = args.api_model

    cap_mode = args.caption_mode
    cap_cmd = [sys.executable, 'tools/captioner.py', '--root', dataset_root, '--out', str(captions_json), '--mode', cap_mode, '--subset', args.subset, '--split', 'query']
    if cap_mode == 'api':
        if args.prompt_file:
            cap_cmd += ['--api_prompt_file', args.prompt_file]
        elif args.prompt:
            cap_cmd += ['--api_prompt', args.prompt]
        if args.api_url: cap_cmd += ['--api_url', args.api_url]
        if args.api_key: cap_cmd += ['--api_key', args.api_key]
        if args.api_model: cap_cmd += ['--api_model', args.api_model]
    elif cap_mode == 'clip_attr':
        cap_cmd += ['--clip_model', args.clip_model, '--clip_device', args.clip_device]
    # pass subset count
    cap_cmd += ['--subset-count', str(args.subset_count)]
    rc_cap, cap_out, t_cap = run_cmd(cap_cmd, cwd=str(workdir))
    Path(outdir / 'logs' / 'caption.stdout.txt').write_text(cap_out, encoding='utf-8')

    # derive query IDs from captions
    try:
        caps = json.loads(Path(captions_json).read_text(encoding='utf-8'))
        q_ids = [str(name.split('_')[0]) for name in caps.keys()]
        np.save(q_ids_path, np.array(q_ids, dtype=object))
        print(f"Saved query IDs: {q_ids_path} ({len(q_ids)})")
    except Exception as e:
        print(f"Failed to build query IDs: {e}")

    # 2) Embed text
    txt_cmd = [sys.executable, 'tools/embed_text.py', '--captions', str(captions_json), '--out', str(text_embeds), '--backend', args.text_backend, '--device', args.device, '--batch-size', str(args.batch_size_text)]
    rc_txt, txt_out, t_txt = run_cmd(txt_cmd, cwd=str(workdir))
    Path(outdir / 'logs' / 'embed_text.stdout.txt').write_text(txt_out, encoding='utf-8')

    # 3) Embed images (gallery) – restrict to subset for examples and log paths
    img_cmd = [sys.executable, 'tools/embed_image.py', '--root', dataset_root, '--out', str(img_embeds), '--split', 'bounding_box_test', '--backend', args.img_backend, '--device', args.device, '--batch-size', str(args.batch_size_image), '--subset', args.subset, '--filter-ids', str(q_ids_path), '--paths-out', str(g_paths_txt), '--subset-count', str(args.subset_count)]
    rc_img, img_out, t_img = run_cmd(img_cmd, cwd=str(workdir))
    Path(outdir / 'logs' / 'embed_image.stdout.txt').write_text(img_out, encoding='utf-8')

    # derive gallery IDs from logged paths if available, else fallback to split scan
    try:
        g_names = []
        if g_paths_txt.exists():
            paths = [line.strip() for line in g_paths_txt.read_text(encoding='utf-8').splitlines() if line.strip()]
            g_names = [Path(p).name for p in paths]
        else:
            gallery_dir = Path(dataset_root) / 'bounding_box_test'
            g_names = [p.name for p in sorted(gallery_dir.glob('*.jpg'))]
        g_ids = [str(n.split('_')[0]) for n in g_names]
        np.save(g_ids_path, np.array(g_ids, dtype=object))
        print(f"Saved gallery IDs: {g_ids_path} ({len(g_ids)})")
    except Exception as e:
        print(f"Failed to build gallery IDs: {e}")

    # 4) Compute similarity and save (optional convenience)
    try:
        T = np.load(text_embeds)
        I = np.load(img_embeds)
        # both expected to be L2-normalized; cosine == dot
        S = T @ I.T
        np.save(sim_npy, S.astype(np.float32))
    except Exception as e:
        S = None
        sim_err = str(e)
    else:
        sim_err = ''

    # 5) Evaluate retrieval
    eval_cmd = [sys.executable, 'tools/retrieve_eval.py', '--text', str(text_embeds), '--img', str(img_embeds), '--out', str(metrics_json), '--query-ids', str(q_ids_path), '--gallery-ids', str(g_ids_path)]
    if args.re_ranking:
        eval_cmd += ['--re_ranking']
    rc_eval, eval_out, t_eval = run_cmd(eval_cmd, cwd=str(workdir))
    Path(outdir / 'logs' / 'eval.stdout.txt').write_text(eval_out, encoding='utf-8')

    # Summarize
    metrics = {}
    try:
        metrics = json.loads(Path(metrics_json).read_text(encoding='utf-8'))
    except Exception:
        pass

    summary = {
        'exp_id': exp_id,
        'dataset_root': dataset_root,
        'subset': args.subset,
        'backends': {'text': args.text_backend, 'image': args.img_backend},
        'caption': {'mode': args.caption_mode},
        'api': {'url': args.api_url or os.environ.get('CAPTION_API_URL',''), 'model': args.api_model or os.environ.get('CAPTION_API_MODEL','')},
        'clip': {'model': args.clip_model, 'device': args.clip_device},
        'eval_opts': {'re_ranking': bool(args.re_ranking)},
        'artifacts': {
            'captions': str(captions_json),
            'text_embeds': str(text_embeds),
            'img_embeds': str(img_embeds),
            'similarity': str(sim_npy),
            'metrics': str(metrics_json),
            'query_ids': str(q_ids_path),
            'gallery_ids': str(g_ids_path),
            'gallery_paths': str(g_paths_txt)
        },
        'elapsed_s': {'caption': t_cap, 'text_embed': t_txt, 'image_embed': t_img, 'eval': t_eval},
        'return_codes': {'caption': rc_cap, 'text_embed': rc_txt, 'image_embed': rc_img, 'eval': rc_eval},
        'metrics': metrics,
        'commands': {
            'caption': cap_cmd,
            'embed_text': txt_cmd,
            'embed_image': img_cmd,
            'eval': eval_cmd
        }
    }

    # Thresholds and caption mode detection
    r1_thr = float(args.r1_thr)
    map_thr = float(args.map_thr)
    if summary.get('caption', {}).get('mode') == 'api':
        api_url_val = summary['api']['url']
        caption_api_mode = 'stub' if (api_url_val.startswith('http://127.0.0.1') or api_url_val.startswith('http://localhost')) else 'remote'
    else:
        caption_api_mode = 'local'
    r1 = float(metrics.get('rank1', 0.0))
    mp = float(metrics.get('mAP', 0.0))
    gate_ok = (r1 >= r1_thr) and (mp >= map_thr)

    # Write run log (human-readable)
    lines = []
    lines.append(f"# Experiment {exp_id}\n")
    lines.append(f"- dataset_root: `{dataset_root}`\n")
    lines.append(f"- subset: `{args.subset}`\n")
    lines.append(f"- subset_count: `{args.subset_count}`\n")
    lines.append(f"- text_backend: `{args.text_backend}`; image_backend: `{args.img_backend}`\n")
    lines.append(f"- caption_mode: `{args.caption_mode}`\n")
    lines.append(f"- api_url: `{summary['api']['url']}`; api_model: `{summary['api']['model']}`\n")
    lines.append(f"- caption_api_mode: `{caption_api_mode}`\n")
    lines.append(f"- clip_caption: model=`{args.clip_model}`, device=`{args.clip_device}`\n")
    lines.append(f"- re_ranking: `{args.re_ranking}`\n")
    lines.append(f"- artifacts:\n")
    for k,v in summary['artifacts'].items():
        lines.append(f"  - {k}: `{v}`\n")
    lines.append(f"- elapsed_s: {summary['elapsed_s']}\n")
    lines.append(f"- return_codes: {summary['return_codes']}\n")
    if metrics:
        lines.append(f"- metrics: rank1={metrics.get('rank1')}, mAP={metrics.get('mAP')}, n_query={metrics.get('n_query')}, n_gallery={metrics.get('n_gallery')}\n")
    lines.append(f"- thresholds: rank1>={r1_thr}, mAP>={map_thr}\n")
    lines.append(f"- gate: {'PASS' if gate_ok else 'FAIL'}\n")
    if S is None and sim_err:
        lines.append(f"- similarity: ERROR `{sim_err}`\n")
    Path(run_log).write_text(''.join(lines), encoding='utf-8')

    # Also dump JSON summary and copy logs
    summary['thresholds'] = {'rank1': r1_thr, 'mAP': map_thr}
    summary['gate'] = 'PASS' if gate_ok else 'FAIL'
    summary['caption_api_mode'] = caption_api_mode
    Path(outdir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"[OK] Experiment {exp_id} completed. See {run_log}")

if __name__ == '__main__':
    main()