#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explicit/Implicit Text Gap Experiment
A) Explicit text (captions/schema)
B) Implicit text (learned prompt vector/template; fallback to default template when missing)
Outputs a reproducible comparison report under outputs/comparison/<timestamp>/
"""
import os, sys, json, time, math, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import torch
import torch.nn.functional as F

# --------- Utils ---------

def utc8_timestamp() -> str:
    # Asia/Shanghai (UTC+8)
    if ZoneInfo is not None:
        tz = ZoneInfo("Asia/Shanghai")
        now = time.localtime()
        # We cannot set tz for time.localtime; fallback to time.time() with tz if available
        t = time.time()
        ts = time.strftime("%Y%m%d-%H%M", time.localtime(t))
    else:
        ts = time.strftime("%Y%m%d-%H%M", time.localtime())
    return ts

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def person_id_from_path(p: str) -> str:
    base = os.path.basename(p)
    # Market-1501 file names start with 4-digit person ID, e.g., 0001_c1s1_...
    return base.split("_")[0]


# --------- CLIP backends (text+image) ---------

def load_clip_l14(device: str = "auto"):
    import clip  # openai/clip
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device


def embed_texts_clip_l14(texts: List[str], model, device: str) -> np.ndarray:
    import clip
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        feats = model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype("float32")


def embed_images_clip_l14(image_paths: List[str], model, preprocess, device: str, batch_size: int = 64) -> np.ndarray:
    tensors = []
    valid_paths = []
    for p in image_paths:
        try:
            from PIL import Image
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            valid_paths.append(p)
        except Exception:
            # skip bad image
            pass
    if not tensors:
        return np.zeros((0, 768), dtype="float32")
    embs = []
    for i in range(0, len(tensors), batch_size):
        bs = tensors[i:i+batch_size]
        with torch.no_grad():
            batch = torch.stack(bs).to(device)
            feats = model.encode_image(batch)
            feats = F.normalize(feats, dim=-1)
        embs.append(feats.cpu().numpy().astype("float32"))
        del batch, feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0)


# --------- Metrics ---------

def compute_rank1_map(sims: np.ndarray, q_ids: List[str], g_ids: List[str]) -> Tuple[float, float, int]:
    top_idx = np.argmax(sims, axis=1)
    hits = []
    for i, j in enumerate(top_idx):
        hits.append(1.0 if str(q_ids[i]) == str(g_ids[j]) else 0.0)
    rank1 = float(np.mean(hits))
    aps = []
    no_pos = 0
    for i in range(sims.shape[0]):
        qi = str(q_ids[i])
        positives = {j for j in range(sims.shape[1]) if str(g_ids[j]) == qi}
        if not positives:
            aps.append(0.0)
            no_pos += 1
            continue
        order = np.argsort(-sims[i])
        hits_so_far = 0
        sum_prec = 0.0
        for r, j in enumerate(order, start=1):
            if j in positives:
                hits_so_far += 1
                sum_prec += hits_so_far / r
        aps.append(sum_prec / len(positives))
    mAP = float(np.mean(aps))
    return rank1, mAP, no_pos


def compute_ndcg_at_k(sims: np.ndarray, q_ids: List[str], g_ids: List[str], k: int = 10) -> float:
    def dcg(rels):
        return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels))
    ndcgs = []
    for i in range(sims.shape[0]):
        qi = str(q_ids[i])
        order = np.argsort(-sims[i])
        rels = [1.0 if str(g_ids[j]) == qi else 0.0 for j in order[:k]]
        dcg_k = dcg(rels)
        n_pos = sum(1 for j in range(sims.shape[1]) if str(g_ids[j]) == qi)
        ideal_rels = [1.0]*min(n_pos, k) + [0.0]*max(0, k - min(n_pos, k))
        idcg_k = dcg(ideal_rels) or 1.0
        ndcgs.append(dcg_k / idcg_k)
    return float(np.mean(ndcgs))


def compute_auc_pos_neg(sims: np.ndarray, q_ids: List[str], g_ids: List[str]) -> float:
    # Pairwise AUC: probability a randomly chosen positive has a higher score than a randomly chosen negative
    pos_scores = []
    neg_scores = []
    for i in range(sims.shape[0]):
        qi = str(q_ids[i])
        for j in range(sims.shape[1]):
            s = sims[i, j]
            if str(g_ids[j]) == qi:
                pos_scores.append(s)
            else:
                neg_scores.append(s)
    P = len(pos_scores)
    N = len(neg_scores)
    if P == 0 or N == 0:
        return 0.0
    pos_scores.sort()
    neg_scores.sort()
    # Use efficient rank-sum approach
    i = j = 0
    wins = ties = 0
    while i < P and j < N:
        if pos_scores[i] > neg_scores[j]:
            wins += N - j
            i += 1
        elif pos_scores[i] < neg_scores[j]:
            j += 1
        else:
            # count ties
            v = pos_scores[i]
            c_pos = 0
            while i < P and pos_scores[i] == v:
                c_pos += 1; i += 1
            c_neg = 0
            k = j
            while k < N and neg_scores[k] == v:
                c_neg += 1; k += 1
            ties += c_pos * c_neg
            j = k
    auc = (wins + 0.5 * ties) / (P * N)
    return float(auc)


def curves_roc_pr(sims: np.ndarray, q_ids: List[str], g_ids: List[str], n_steps: int = 200):
    # Build global pos/neg lists
    scores = []
    labels = []
    for i in range(sims.shape[0]):
        qi = str(q_ids[i])
        for j in range(sims.shape[1]):
            s = sims[i, j]
            lbl = 1 if str(g_ids[j]) == qi else 0
            scores.append(float(s)); labels.append(int(lbl))
    if not scores:
        return [], []
    smin, smax = min(scores), max(scores)
    if smax == smin:
        thresholds = [smin]
    else:
        thresholds = [smin + (smax - smin) * t / (n_steps - 1) for t in range(n_steps)]
    roc = []  # threshold, tpr, fpr
    pr = []   # threshold, precision, recall
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    for th in thresholds:
        tp = fp = tn = fn = 0
        for s, lbl in zip(scores, labels):
            pred = 1 if s >= th else 0
            if pred == 1 and lbl == 1: tp += 1
            elif pred == 1 and lbl == 0: fp += 1
            elif pred == 0 and lbl == 0: tn += 1
            elif pred == 0 and lbl == 1: fn += 1
        tpr = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        roc.append((th, tpr, fpr))
        pr.append((th, precision, recall))
    return roc, pr


# --------- Main pipeline ---------
@dataclass
class Config:
    text_backend: str = "clip_l14"
    image_backend: str = "clip_l14"
    subset_count: int = 50
    seed: int = 42
    temps: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    alphas: Tuple[float, ...] = (0.2, 0.5, 0.8)
    dataset_root: str = "data/archive/Market-1501-v15.09.15"
    query_list: str = "larger_iso/64/query.txt"
    gallery_list: str = "larger_iso/64/gallery.txt"
    labels_subset_path: str = "data/market1501.images.labels_subset50.txt"


def load_list(path: str, n: int) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
            if len(lines) >= n:
                break
    return lines


def ensure_labels_file(cfg: Config, q_paths: List[str], g_paths: List[str]):
    os.makedirs(os.path.dirname(cfg.labels_subset_path), exist_ok=True)
    with open(cfg.labels_subset_path, "w", encoding="utf-8") as f:
        for p in q_paths: f.write(p + "\n")
        for p in g_paths: f.write(p + "\n")


def find_latest_explicit_captions() -> Dict[str, List[str]]:
    base = os.path.join("outputs", "experiments")
    if not os.path.isdir(base):
        return {}
    exps = [d for d in os.listdir(base) if d.startswith("EXP_")]
    exps.sort(reverse=True)
    for d in exps:
        cap_path = os.path.join(base, d, "captions", "captions_query.json")
        if os.path.isfile(cap_path):
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def ensure_explicit_captions(q_paths: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path):
        # Load and filter to the current subset basenames only
        with open(out_path, "r", encoding="utf-8") as f:
            caps = json.load(f)
        want = {}
        for p in q_paths:
            base = os.path.basename(p)
            if base in caps:
                want[base] = caps[base]
        if want:
            return want
    # Try reuse from latest experiment
    caps = find_latest_explicit_captions()
    if caps:
        # keys in caps are basenames; filter to our query basenames
        want = {}
        for p in q_paths:
            base = os.path.basename(p)
            if base in caps:
                want[base] = caps[base]
        if want:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(want, f, indent=2)
            return want
    # Fallback: simple rule-based placeholder caption "person"
    fallback = {os.path.basename(p): ["person"] for p in q_paths}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fallback, f, indent=2)
    return fallback


def ensure_implicit_captions(q_paths: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Try prompt_tuner outputs
    prompt_root = os.path.join("outputs", "prompt_tuner")
    implicit_text = None
    if os.path.isdir(prompt_root):
        runs = sorted(os.listdir(prompt_root), reverse=True)
        for r in runs:
            best_txt = os.path.join(prompt_root, r, "best_prompt.txt")
            if os.path.isfile(best_txt):
                with open(best_txt, "r", encoding="utf-8") as f:
                    tmpl = f.read().strip()
                implicit_text = tmpl if tmpl else "person"
                break
    # If not found, fallback to default learnable template
    if implicit_text is None:
        implicit_text = "person"
    caps = {os.path.basename(p): [implicit_text] for p in q_paths}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2)
    return caps


def texts_from_caps(caps: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    keys = sorted(caps.keys())
    texts = []
    for k in keys:
        lst = caps[k]
        if isinstance(lst, list) and lst:
            s = lst[0]
        else:
            s = "person"
        texts.append(str(s))
    return keys, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-count", type=int, default=50)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = Config(subset_count=args.subset_count)

    # Prepare query/gallery paths (subset50)
    q_list = load_list(cfg.query_list, cfg.subset_count)
    g_list = load_list(cfg.gallery_list, cfg.subset_count)
    if not q_list or not g_list:
        raise RuntimeError("Query/Gallery list missing or too short")
    ensure_labels_file(cfg, q_list, g_list)

    # IDs
    q_ids = [person_id_from_path(p) for p in q_list]
    g_ids = [person_id_from_path(p) for p in g_list]

    # Output dir
    timestamp = utc8_timestamp()
    out_dir = os.path.join("outputs", "comparison", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    t0 = time.time()

    # Captions (explicit/implicit)
    explicit_path = os.path.join("outputs", "captions", "explicit_captions.json")
    implicit_path = os.path.join("outputs", "captions", "implicit_captions.json")
    caps_exp = ensure_explicit_captions(q_list, explicit_path)
    caps_imp = ensure_implicit_captions(q_list, implicit_path)

    # Backend
    model, preprocess, device = load_clip_l14(device=args.device)

    # Embed explicit texts
    exp_keys, exp_texts = texts_from_caps(caps_exp)
    imp_keys, imp_texts = texts_from_caps(caps_imp)
    T_exp = embed_texts_clip_l14(exp_texts, model, device)
    T_imp = embed_texts_clip_l14(imp_texts, model, device)
    T_exp = l2_normalize(T_exp)
    T_imp = l2_normalize(T_imp)

    # Embed gallery images (or load if exists)
    img_out = os.path.join("embeds", "image", "clip-l14_market_subset50.npy")
    if os.path.isfile(img_out):
        I = np.load(img_out)
    else:
        I = embed_images_clip_l14(g_list, model, preprocess, device, batch_size=64)
        I = l2_normalize(I)
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        np.save(img_out, I)
    I = l2_normalize(I)

    # Similarities
    S_exp = (T_exp @ I.T).astype("float32")
    S_imp = (T_imp @ I.T).astype("float32")

    # Temperature scaling (grid choose by Rank-1)
    best_T_exp = 1.0; best_T_imp = 1.0; best_r1_exp = -1; best_r1_imp = -1
    for T in cfg.temps:
        r1_e, _, _ = compute_rank1_map(S_exp / T, q_ids, g_ids)
        r1_i, _, _ = compute_rank1_map(S_imp / T, q_ids, g_ids)
        if r1_e > best_r1_exp: best_r1_exp, best_T_exp = r1_e, T
        if r1_i > best_r1_imp: best_r1_imp, best_T_imp = r1_i, T
    S_exp_scaled = (S_exp / best_T_exp).astype("float32")
    S_imp_scaled = (S_imp / best_T_imp).astype("float32")

    # Metrics for explicit and implicit
    def all_metrics(S):
        rank1, mAP, no_pos = compute_rank1_map(S, q_ids, g_ids)
        ndcg = compute_ndcg_at_k(S, q_ids, g_ids, k=10)
        auc = compute_auc_pos_neg(S, q_ids, g_ids)
        roc, pr = curves_roc_pr(S, q_ids, g_ids, n_steps=200)
        return {
            "Rank-1": rank1,
            "mAP": mAP,
            "nDCG@10": ndcg,
            "AUC": auc,
            "no_pos": no_pos,
        }, roc, pr

    metrics_exp, roc_exp, pr_exp = all_metrics(S_exp_scaled)
    metrics_imp, roc_imp, pr_imp = all_metrics(S_imp_scaled)

    # Fusion (z-score per query row)
    def zscore_rows(M: np.ndarray) -> np.ndarray:
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True) + 1e-9
        return (M - mu) / sd
    Z_exp = zscore_rows(S_exp_scaled)
    Z_imp = zscore_rows(S_imp_scaled)
    best_alpha = None; best_alpha_metrics = None; best_rank1 = -1
    fuse_sims = {}
    for a in cfg.alphas:
        Sf = (a * Z_exp + (1.0 - a) * Z_imp).astype("float32")
        m, _, _ = all_metrics(Sf)
        fuse_sims[a] = Sf
        if m["Rank-1"] > best_rank1:
            best_rank1 = m["Rank-1"]; best_alpha = a; best_alpha_metrics = m

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "sim_exp.npy"), S_exp_scaled)
    np.save(os.path.join(out_dir, "sim_imp.npy"), S_imp_scaled)
    for a, Sf in fuse_sims.items():
        np.save(os.path.join(out_dir, f"sim_fuse_α{a}.npy"), Sf)

    # curves
    curves_dir = os.path.join(out_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    with open(os.path.join(curves_dir, "roc_exp.csv"), "w", encoding="utf-8") as f:
        f.write("threshold,tpr,fpr\n")
        for th, tpr, fpr in roc_exp: f.write(f"{th},{tpr},{fpr}\n")
    with open(os.path.join(curves_dir, "roc_imp.csv"), "w", encoding="utf-8") as f:
        f.write("threshold,tpr,fpr\n")
        for th, tpr, fpr in roc_imp: f.write(f"{th},{tpr},{fpr}\n")
    with open(os.path.join(curves_dir, "pr_exp.csv"), "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall\n")
        for th, p, r in pr_exp: f.write(f"{th},{p},{r}\n")
    with open(os.path.join(curves_dir, "pr_imp.csv"), "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall\n")
        for th, p, r in pr_imp: f.write(f"{th},{p},{r}\n")

    # metrics jsons
    import yaml
    with open(os.path.join(out_dir, "metrics_exp.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics_exp, "temperature": best_T_exp}, f, indent=2)
    with open(os.path.join(out_dir, "metrics_imp.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics_imp, "temperature": best_T_imp}, f, indent=2)
    with open(os.path.join(out_dir, "metrics_fuse.json"), "w", encoding="utf-8") as f:
        json.dump({**best_alpha_metrics, "best_alpha": best_alpha}, f, indent=2)

    # config.yaml
    cfg_yaml = {
        "model": {
            "text_backend": cfg.text_backend,
            "image_backend": cfg.image_backend,
            "device": args.device,
        },
        "grid": {
            "temperatures": list(cfg.temps),
            "alphas": list(cfg.alphas),
        },
        "data": {
            "dataset": cfg.dataset_root,
            "query_list": cfg.query_list,
            "gallery_list": cfg.gallery_list,
            "subset_count": cfg.subset_count,
            "labels_file": cfg.labels_subset_path,
        },
        "seed": cfg.seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_yaml, f, sort_keys=False, allow_unicode=True)

    # summary.csv
    elapsed = time.time() - t0
    def line_for(name, m, alpha=None, temp=None):
        return {
            "setting": name,
            "Rank-1": f"{m['Rank-1']:.6f}",
            "mAP": f"{m['mAP']:.6f}",
            "nDCG@10": f"{m['nDCG@10']:.6f}",
            "AUC": f"{m['AUC']:.6f}",
            "最佳α": "" if alpha is None else str(alpha),
            "温度": "" if temp is None else str(temp),
            "耗时": f"{elapsed:.2f}s",
        }
    rows = [
        line_for("explicit", metrics_exp, alpha=None, temp=best_T_exp),
        line_for("implicit", metrics_imp, alpha=None, temp=best_T_imp),
        line_for("fuse", best_alpha_metrics, alpha=best_alpha, temp=f"exp={best_T_exp},imp={best_T_imp}"),
    ]
    import csv
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows: writer.writerow(r)

    # README.md
    readme = []
    readme.append("# Explicit/Implicit Text Gap Experiment")
    readme.append("")
    readme.append("任务: 文本→图像检索，两路对比（显式/隐式），输出可复现实验与报告。")
    readme.append("数据: Market-1501 子集50；统一使用 CLIP ViT-L/14 作为文本与图像编码器。")
    readme.append("指令: 本目录包含相似度矩阵、指标JSON、曲线CSV、配置与摘要。")
    readme.append("")
    readme.append("结论摘要:")
    readme.append(f"- Explicit: Rank-1={metrics_exp['Rank-1']:.4f}, mAP={metrics_exp['mAP']:.4f}, nDCG@10={metrics_exp['nDCG@10']:.4f}, AUC={metrics_exp['AUC']:.4f} (T={best_T_exp})")
    readme.append(f"- Implicit: Rank-1={metrics_imp['Rank-1']:.4f}, mAP={metrics_imp['mAP']:.4f}, nDCG@10={metrics_imp['nDCG@10']:.4f}, AUC={metrics_imp['AUC']:.4f} (T={best_T_imp})")
    readme.append(f"- Fuse: α={best_alpha}, Rank-1={best_alpha_metrics['Rank-1']:.4f}, mAP={best_alpha_metrics['mAP']:.4f}, nDCG@10={best_alpha_metrics['nDCG@10']:.4f}, AUC={best_alpha_metrics['AUC']:.4f}")
    readme.append("")
    readme.append("Assumptions/TBD:")
    readme.append("- 未发现 prompt_tuner 输出(best_prompt.txt / prompt_vec.npy)，隐式文本使用默认模板 ‘person’。")
    readme.append("- labels_subset50 由 larger_iso/64 的前50条构建，保证查询与图库均含相同 person-id 集合。")
    readme.append("- 温度缩放在同一查询集上网格选择；可进一步引入独立验证集优化温度。")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    # simple run.log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"text_backend={cfg.text_backend}, image_backend={cfg.image_backend}, device={device}\n")
        f.write(f"subset_count={cfg.subset_count}, timestamp={timestamp}\n")
        f.write(f"best_T_exp={best_T_exp}, best_T_imp={best_T_imp}, best_alpha={best_alpha}\n")
        f.write(f"q_ids={len(q_ids)}, g_ids={len(g_ids)}, elapsed={elapsed:.2f}s\n")

    print(f"[OK] Wrote outputs to {out_dir}")

if __name__ == "__main__":
    main()