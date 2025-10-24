#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-entry pipeline for v4.1-soft A→B→C
- A: soft color consistency
- B: tri-window explicit grid scan (soft)
- C: LR fusion (consistency + best tri-window explicit + tri-window attr)

Constraints:
- CLIP ViT-L/14, center crop, L2; global z-score only
- If inputs missing, write ERRORS.md but continue subsequent steps
- Each subdir must write logs and config.yaml

Overview row schema:
setting, Rank-1, mAP, nDCG@10, fusion, rerank, gamma, triwin_w_exp, triwin_w_attr, known_rate, top1_mismatch, seed, ts
"""

import argparse
import json
import time
from pathlib import Path
import importlib
import sys
import csv
import traceback
import numpy as np
import platform
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "comparison"

# -------- helpers --------

def parse_tuple_list(s_list):
    """Parse list like ["(0.5,0.3,0.2)", "(0.6,0.3,0.1)"] into list of tuples(float)."""
    res = []
    for s in s_list:
        s = s.strip()
        if not (s.startswith("(") and s.endswith(")")):
            raise ValueError(f"Bad tuple format: {s}")
        parts = s[1:-1].split(",")
        res.append(tuple(float(p.strip()) for p in parts))
    return res


def ensure_rel_to_root(p_str):
    p = Path(p_str)
    return (ROOT / p) if not p.is_absolute() else p


def fixed_ts_tag():
    return time.strftime("%Y%m%d-%H%M%S")


def write_errors_md(dir_path: Path, errors: list):
    try:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "ERRORS.md").write_text("\n".join(errors), encoding="utf-8")
    except Exception:
        # best-effort; do not raise
        pass


def read_known_rate_from_A(A_dir: Path) -> float:
    try:
        q = json.loads((A_dir / "query_color_labels.json").read_text(encoding="utf-8"))
        g = json.loads((A_dir / "gallery_color_labels.json").read_text(encoding="utf-8"))
        def _rate(labels):
            total = len(labels)
            # prefer 'unknown' flag; fallback to 'known'
            unknown = sum(1 for x in labels.values() if x and x.get("unknown", False))
            if unknown == 0 and all(isinstance(x, dict) for x in labels.values()):
                known = sum(1 for x in labels.values() if x and x.get("known", False))
                return known / max(total, 1)
            return (total - unknown) / max(total, 1)
        return round(( _rate(q) + _rate(g) ) / 2.0, 4)
    except Exception:
        # fallback: try summary.csv
        try:
            with (A_dir / "summary.csv").open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows and "known_rate" in rows[-1]:
                    val = rows[-1]["known_rate"]
                    return float(val) if val else -1.0
        except Exception:
            pass
        return -1.0


def read_metrics_from_C(C_dir: Path):
    rank1 = map_ = ndcg = -1.0
    top1_mismatch = -1.0
    try:
        m = json.loads((C_dir / "metrics_fuse_lr.json").read_text(encoding="utf-8"))
        rank1 = float(m.get("Rank-1", -1))
        map_ = float(m.get("mAP", -1))
        ndcg = float(m.get("nDCG@10", -1))
    except Exception:
        pass
    try:
        # summary.csv may include top1_mismatch
        with (C_dir / "summary.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # use the last row if multiple
            rows = list(reader)
            if rows:
                row = rows[-1]
                if "top1_mismatch" in row:
                    top1_mismatch = float(row["top1_mismatch"]) if row["top1_mismatch"] else -1.0
    except Exception:
        pass
    return rank1, map_, ndcg, top1_mismatch


def append_overview(ts: str, setting: str, fusion: str, rerank: str, gamma_str: str,
                    triwin_w_exp: str, triwin_w_attr: str, known_rate: float,
                    top1_mismatch: float, seed: int,
                    rank1: float, map_: float, ndcg: float):
    overview_path = OUT_ROOT / f"{ts}_overview.csv"
    header = [
        "setting", "Rank-1", "mAP", "nDCG@10", "fusion", "rerank",
        "gamma", "triwin_w_exp", "triwin_w_attr", "known_rate",
        "top1_mismatch", "seed", "ts"
    ]
    row = {
        "setting": setting,
        "Rank-1": f"{rank1:.4f}" if rank1 >= 0 else "",
        "mAP": f"{map_:.4f}" if map_ >= 0 else "",
        "nDCG@10": f"{ndcg:.4f}" if ndcg >= 0 else "",
        "fusion": fusion,
        "rerank": rerank,
        "gamma": gamma_str,
        "triwin_w_exp": triwin_w_exp,
        "triwin_w_attr": triwin_w_attr,
        "known_rate": f"{known_rate:.4f}" if known_rate >= 0 else "",
        "top1_mismatch": f"{top1_mismatch:.4f}" if top1_mismatch >= 0 else "",
        "seed": str(seed),
        "ts": ts,
    }
    exists = overview_path.exists()
    with overview_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -------- main pipeline --------

def main():
    parser = argparse.ArgumentParser(description="Run v4.1-soft pipeline A→B→C")
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--embeds", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--gamma", nargs="+", type=float, required=True)
    parser.add_argument("--tau", nargs="+", type=float, required=True)
    parser.add_argument("--theta", nargs="+", type=float, required=True)
    parser.add_argument("--delta_deg", nargs="+", type=float, required=True)
    parser.add_argument("--triwin-exp", nargs="+", required=True)
    parser.add_argument("--triwin-attr", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ts", type=str, default=None, help="override timestamp to use precomputed outputs")
    parser.add_argument("--only-stamp", action="store_true", help="skip running steps; only stamp and overview")
    args = parser.parse_args()

    # Local helper defs to ensure availability before main() runs
    def _compute_sha256(p: Path) -> str:
        try:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def _read_triwin_w_attr_from_C(C_dir: Path) -> str:
        try:
            text = (C_dir / "config.yaml").read_text(encoding="utf-8")
            try:
                cfg = json.loads(text)
                v = cfg.get("triwin_w_attr", "")
                if v:
                    return str(v)
            except Exception:
                pass
            import re
            m = re.search(r"triwin_w_attr\s*[:=]\s*([\w\d_]+)", text)
            if m:
                return m.group(1)
        except Exception:
            pass
        return ""

    def _read_best_params_from_A(A_dir: Path):
        tau = theta = delta = None
        try:
            text = (A_dir / "config.yaml").read_text(encoding="utf-8")
            try:
                cfg = json.loads(text)
                best = cfg.get("best", {})
                tau = best.get("tau")
                theta = best.get("theta")
                delta = best.get("delta")
            except Exception:
                pass
            if tau is None or theta is None or delta is None:
                import re
                m_tau = re.search(r"tau\s*[:=]\s*([0-9.]+)", text)
                m_theta = re.search(r"theta\s*[:=]\s*([0-9.]+)", text)
                m_delta = re.search(r"delta\s*[:=]\s*([0-9.]+)", text)
                tau = tau or (float(m_tau.group(1)) if m_tau else None)
                theta = theta or (float(m_theta.group(1)) if m_theta else None)
                delta = delta or (float(m_delta.group(1)) if m_delta else None)
        except Exception:
            pass
        return tau, theta, delta

    def _write_env_lock(seed: int, ts: str, embeds_path: Path):
        info = {
            "ts": ts,
            "seed": seed,
            "python_version": sys.version,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "packages": {},
            "cuda": {},
            "clip_model_name": "CLIP ViT-L/14",
            "embeds_path": str(embeds_path),
            "embeds_sha256": _compute_sha256(embeds_path),
        }
        try:
            import torch
            info["packages"]["torch"] = getattr(torch, "__version__", "")
            info["cuda"]["available"] = bool(getattr(torch.cuda, "is_available", lambda: False)())
            info["cuda"]["version"] = getattr(torch.version, "cuda", None)
            try:
                info["cuda"]["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)()
            except Exception:
                info["cuda"]["cudnn_version"] = None
        except Exception:
            info["packages"]["torch"] = None
        try:
            info["packages"]["numpy"] = __import__("numpy").__version__
        except Exception:
            info["packages"]["numpy"] = None
        try:
            info["packages"]["scikit_learn"] = __import__("sklearn").__version__
        except Exception:
            info["packages"]["scikit_learn"] = None
        try:
            (ROOT / "ENV_LOCK.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _stamp_config_baseline(cfg_path: Path, ts: str):
        try:
            text = cfg_path.read_text(encoding="utf-8")
            if "baseline:" in text:
                return
            cfg_path.write_text(text + f"\nbaseline: baseline@{ts}\n", encoding="utf-8")
        except Exception:
            pass

    def _stamp_baseline(ts: str, seed: int, A_dir: Path, B_dir: Path, C_dir: Path, overview_path: Path,
                        triwin_w_exp: str, triwin_w_attr: str, tau, theta, delta):
        meta = {
            "tag": f"baseline@{ts}",
            "setting": "attrbank_v4_1_lr_soft",
            "seed": seed,
            "triwin_w_exp": triwin_w_exp,
            "triwin_w_attr": triwin_w_attr,
            "tau": tau,
            "theta": theta,
            "delta_deg": delta,
            "overview_csv": str(overview_path),
        }
        for d in [A_dir, B_dir, C_dir]:
            try:
                (d / "baseline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass
        for d in [A_dir, B_dir, C_dir]:
            _stamp_config_baseline(d / "config.yaml", ts)
        try:
            (C_dir / "weights.pkl.baseline").write_text(f"baseline@{ts}\n", encoding="utf-8")
        except Exception:
            pass
        try:
            (C_dir / "calibration.json.baseline").write_text(f"baseline@{ts}\n", encoding="utf-8")
        except Exception:
            pass
        try:
            (overview_path.parent / f"{overview_path.stem}.baseline").write_text(f"baseline@{ts}\n", encoding="utf-8")
        except Exception:
            pass

    def _append_results_md(ts: str, seed: int, rank1: float, map_: float, ndcg: float,
                           known_rate: float, top1_mis: float, w_exp: str, w_attr: str,
                           tau, theta, delta):
        line = (
            f"baseline@{ts} | setting=attrbank_v4_1_lr_soft | seed={seed} | "
            f"R1={rank1:.4f}, mAP={map_:.4f}, nDCG10={ndcg:.4f}, known_rate={known_rate:.4f}, top1_mismatch={top1_mis:.4f} | "
            f"w_exp={w_exp}, w_attr={w_attr}, tau={tau}, theta={theta}, delta={delta}°\n"
        )
        try:
            with (ROOT / "RESULTS.md").open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    ts = args.ts if args.ts else fixed_ts_tag()
    np.random.seed(args.seed)

    # normalize paths
    query_path = ensure_rel_to_root(args.query)
    gallery_path = ensure_rel_to_root(args.gallery)
    embeds_path = ensure_rel_to_root(args.embeds)
    captions_path = ensure_rel_to_root(args.captions)

    # parse weights
    tri_exp_weights = parse_tuple_list(args.triwin_exp)  # list of tuples
    tri_attr_weights = parse_tuple_list(args.triwin_attr)  # list (use first)
    tri_attr_str = args.triwin_attr[0]

    # Step A: soft color consistency
    A_errors = []
    try:
        if not args.only_stamp:
            A = importlib.import_module("tools.run_attrbank_v4_1_colorcons_soft")
            A.ts = lambda: ts
            A.QUERY_LIST = query_path
            A.GALLERY_LIST = gallery_path
            A.EMB_IMAGE_PATH = embeds_path
            if hasattr(A, "CAP_PATH"):
                A.CAP_PATH = captions_path
            A.TAUS = list(args.tau)
            A.THETAS = list(args.theta)
            A.DELTAS = list(args.delta_deg)
            A.GAMMAS = list(args.gamma)
            if hasattr(A, "SEED"):
                A.SEED = args.seed
            A.main()
    except Exception as e:
        A_errors.append("[A] " + str(e))
        A_errors.append(traceback.format_exc())
        write_errors_md(OUT_ROOT / f"{ts}_attrbank_v4_1_colorcons_soft", A_errors)
    A_dir = OUT_ROOT / f"{ts}_attrbank_v4_1_colorcons_soft"

    # Step B: tri-window explicit grid (soft)
    B_errors = []
    best_exp_key = None
    try:
        if not args.only_stamp:
            B = importlib.import_module("tools.run_attrbank_v4_1_triwin_grid")
            B.ts = lambda: ts
            B.QUERY_LIST = query_path
            B.GALLERY_LIST = gallery_path
            B.CAP_PATH = captions_path
            if hasattr(B, "SEED"):
                B.SEED = args.seed
            wsets = {}
            for i, w in enumerate(tri_exp_weights, start=1):
                wsets[f"w{i}"] = list(w)
            B.WSETS = wsets
            B.main()
            B_dir = OUT_ROOT / f"{ts}_attrbank_v4_1_triwin_grid_soft"
            cfg = json.loads((B_dir / "config.yaml").read_text(encoding="utf-8"))
            best_exp_key = cfg.get("best_exp_key", "")
        else:
            # read existing best key
            B_dir = OUT_ROOT / f"{ts}_attrbank_v4_1_triwin_grid_soft"
            if (B_dir / "config.yaml").exists():
                text_b = (B_dir / "config.yaml").read_text(encoding="utf-8")
                try:
                    cfg_b = json.loads(text_b)
                    best_exp_key = cfg_b.get("best_exp_key", "")
                except Exception:
                    import re
                    m = re.search(r"best_exp_key\s*[:=]\s*([\w\d_]+)", text_b)
                    best_exp_key = m.group(1) if m else ""
    except Exception as e:
        B_errors.append("[B] " + str(e))
        B_errors.append(traceback.format_exc())
        write_errors_md(OUT_ROOT / f"{ts}_attrbank_v4_1_triwin_grid_soft", B_errors)

    # Step C: LR fusion
    C_errors = []
    try:
        if not args.only_stamp:
            C = importlib.import_module("tools.run_attrbank_v4_1_lr")
            C.ts = lambda: ts
            C.QUERY_LIST = query_path
            C.GALLERY_LIST = gallery_path
            if hasattr(C, "SEED"):
                C.SEED = args.seed
            C.main()
    except Exception as e:
        C_errors.append("[C] " + str(e))
        C_errors.append(traceback.format_exc())
        write_errors_md(OUT_ROOT / f"{ts}_attrbank_v4_1_lr_soft", C_errors)

    # Overview aggregation
    C_dir = OUT_ROOT / f"{ts}_attrbank_v4_1_lr_soft"
    rank1, map_, ndcg, top1_mis = read_metrics_from_C(C_dir)
    known_rate = read_known_rate_from_A(A_dir)

    # gamma is not used directly by LR fusion; record as '-'
    gamma_str = "-"
    triwin_w_exp = best_exp_key or ""
    triwin_w_attr = _read_triwin_w_attr_from_C(C_dir) or (args.triwin_attr[0] if args.triwin_attr else "")
    setting = "attrbank_v4_1_lr_soft"
    fusion = "LR+consistency"
    rerank = "none"

    append_overview(
        ts=ts,
        setting=setting,
        fusion=fusion,
        rerank=rerank,
        gamma_str=gamma_str,
        triwin_w_exp=triwin_w_exp,
        triwin_w_attr=tri_attr_str,
        known_rate=known_rate,
        top1_mismatch=top1_mis,
        seed=args.seed,
        rank1=rank1,
        map_=map_,
        ndcg=ndcg,
    )

    # Environment lock
    _write_env_lock(args.seed, ts, embeds_path)
    
    # Baseline stamping
    tau, theta, delta = _read_best_params_from_A(A_dir)
    overview_path = OUT_ROOT / f"{ts}_overview.csv"
    _stamp_baseline(ts, args.seed, A_dir, OUT_ROOT / f"{ts}_attrbank_v4_1_triwin_grid_soft", C_dir,
                    overview_path, triwin_w_exp, (triwin_w_attr or tri_attr_str), tau, theta, delta)
    
    # RESULTS.md append
    _append_results_md(ts, args.seed, rank1, map_, ndcg, known_rate, top1_mis,
                       triwin_w_exp, (triwin_w_attr or tri_attr_str), tau, theta, delta)
    
    # Acceptance gate reporting (stdout only) and hard guard
    ok = (known_rate >= 0.30 and top1_mis <= 0.50 and map_ >= 0.120 and ndcg >= 0.150)
    payload = {
        "ts": ts,
        "Rank-1": rank1,
        "mAP": map_,
        "nDCG@10": ndcg,
        "known_rate": known_rate,
        "top1_mismatch": top1_mis,
        "triwin_w_exp": triwin_w_exp,
        "triwin_w_attr": (triwin_w_attr or tri_attr_str),
        "seed": args.seed,
        "acceptance_pass": bool(ok),
        "errors": {
            "A": bool(A_errors),
            "B": bool(B_errors),
            "C": bool(C_errors)
        }
    }
    print(json.dumps(payload, ensure_ascii=False))
    if not ok and not args.only_stamp:
        sys.exit(1)


# Entrypoint moved below helper defs
if __name__ == "__main__":
    sys.path.append(str(ROOT))
    main()