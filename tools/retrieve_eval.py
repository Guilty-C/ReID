#!/usr/bin/env python
import argparse, json, numpy as np, os

def load_embeddings(file_path):
    """加载嵌入文件，支持标准numpy格式和内存映射格式"""
    import os
    import numpy as np
    
    # 首先尝试标准numpy加载
    try:
        return np.load(file_path)
    except:
        pass
    
    # 如果标准加载失败，尝试内存映射
    file_size = os.path.getsize(file_path)
    
    # 检查文件头信息
    with open(file_path, 'rb') as f:
        try:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            # 如果是标准numpy格式但np.load失败，使用内存映射
            return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
        except:
            # 如果是原始内存映射格式，推断形状
            n_embeddings = file_size // (768 * 4)  # float32 = 4 bytes
            if n_embeddings * 768 * 4 == file_size:
                return np.memmap(file_path, dtype='float32', mode='r', shape=(n_embeddings, 768))
            else:
                raise ValueError(f"Cannot load embeddings from {file_path}")

def normalize_embeddings():
    """归一化嵌入向量并保存"""
    import numpy as np
    
    # 加载文本嵌入（标准numpy格式）
    text_embeddings = np.load(r'outputs/text_embeddings_full.npy')
    
    # 加载图像嵌入（内存映射格式）
    img_file_size = os.path.getsize(r'outputs/embeddings_full.npy')
    n_embeddings = img_file_size // (768 * 4)  # float32 = 4 bytes
    image_embeddings = np.memmap(r'outputs/embeddings_full.npy', dtype='float32', mode='r', shape=(n_embeddings, 768))
    
    # 归一化处理
    text_embeddings_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-9)
    image_embeddings_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-9)
    
    # 保存归一化后的嵌入
    np.save(r'outputs/embeds/text_embeds_norm.npy', text_embeddings_norm)
    np.save(r'outputs/embeds/img_embeds_norm.npy', image_embeddings_norm)
    
    print(f'[OK] normalized text: {text_embeddings_norm.shape}, image: {image_embeddings_norm.shape}')
    return text_embeddings_norm, image_embeddings_norm

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """重排序算法实现"""
    # 简化版本的重排序
    original_dist = np.copy(q_g_dist)
    q_g_dist = q_g_dist / np.max(q_g_dist)
    q_q_dist = q_q_dist / np.max(q_q_dist)
    g_g_dist = g_g_dist / np.max(g_g_dist)
    
    # 计算重排序距离
    re_dist = np.zeros_like(q_g_dist)
    for i in range(q_g_dist.shape[0]):
        for j in range(q_g_dist.shape[1]):
            re_dist[i, j] = (1 - lambda_value) * q_g_dist[i, j] + \
                           lambda_value * (q_q_dist[i, i] + g_g_dist[j, j]) / 2
    
    return re_dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--re_ranking", action="store_true", help="Enable re-ranking")
    ap.add_argument("--query-ids", dest="query_ids", required=False, help="Path to query IDs (.npy or .csv)")
    ap.add_argument("--gallery-ids", dest="gallery_ids", required=False, help="Path to gallery IDs (.npy or .csv)")
    args = ap.parse_args()

    def load_ids(path, expected_len=None):
        import csv
        if not path:
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            ids = [str(x) for x in arr.tolist()]
            return np.array(ids, dtype=object)
        # CSV or TXT
        ids = []
        with open(path, "r", encoding="utf-8") as f:
            try:
                reader = csv.reader(f)
                rows = list(reader)
            except Exception:
                rows = [line.strip().split(",") for line in f.read().splitlines()]
        if rows:
            header = rows[0]
            if any(h.lower() == "id" for h in header):
                idx = [i for i, h in enumerate(header) if h.lower() == "id"][0]
                for r in rows[1:]:
                    if idx < len(r):
                        ids.append(str(r[idx]).strip())
            elif len(header) == 1 and not header[0].strip() or (len(rows) > 1 and len(rows[0]) == 1):
                for r in rows:
                    if r:
                        ids.append(str(r[0]).strip())
            else:
                # heuristics: try second column
                for r in rows[1:] if any(h.lower() in ("image_path", "path", "img") for h in header) else rows:
                    if len(r) >= 2:
                        ids.append(str(r[1]).strip())
                    elif r:
                        ids.append(str(r[0]).strip())
        if expected_len is not None and len(ids) != expected_len:
            print(f"[WARN] IDs length mismatch {len(ids)} vs expected {expected_len}")
        return np.array(ids, dtype=object)

    try:
        text_embeddings = load_embeddings(args.text)
        image_embeddings = load_embeddings(args.img)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Image embeddings shape: {image_embeddings.shape}")

        # full similarity matrix
        sims = text_embeddings @ image_embeddings.T
        print(f"Similarity matrix shape: {sims.shape}")

        # optional re-ranking
        if args.re_ranking:
            print("Applying re-ranking...")
            q_q = text_embeddings @ text_embeddings.T
            g_g = image_embeddings @ image_embeddings.T
            sims = re_ranking(sims, q_q, g_g)
            print("Re-ranking completed")

        # validate finite values
        if not np.all(np.isfinite(sims)):
            raise ValueError("Similarity contains NaN/Inf")

        q_ids = load_ids(args.query_ids, expected_len=text_embeddings.shape[0]) if args.query_ids else None
        g_ids = load_ids(args.gallery_ids, expected_len=image_embeddings.shape[0]) if args.gallery_ids else None

        no_pos = 0
        if q_ids is not None and g_ids is not None:
            # Rank-1 by ID match of top-1 gallery per query
            top_idx = np.argmax(sims, axis=1)
            hits = []
            for i, j in enumerate(top_idx):
                hits.append(1.0 if str(q_ids[i]) == str(g_ids[j]) else 0.0)
            rank1 = float(np.mean(hits))

            # mAP computation
            aps = []
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
        else:
            print("[WARN] Missing IDs; falling back to diagonal Rank-1 and mAP=0.")
            diag_scores = np.diag(sims) if sims.shape[0] == sims.shape[1] else np.zeros((sims.shape[0],))
            mask = np.ones_like(sims, dtype=bool)
            if sims.shape[0] == sims.shape[1]:
                np.fill_diagonal(mask, False)
                max_other = np.max(np.where(mask, sims, -np.inf), axis=1)
                rank1 = float(np.mean(diag_scores > max_other))
            else:
                rank1 = 0.0
            mAP = 0.0

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "rank1": float(rank1),
                "mAP": float(mAP),
                "n_query": int(text_embeddings.shape[0]),
                "n_gallery": int(image_embeddings.shape[0]),
                "no_pos": int(no_pos),
                "similarity_shape": [int(sims.shape[0]), int(sims.shape[1])]
            }, f, indent=2)
        print(f"Evaluation completed: Rank-1 = {rank1:.4f}, mAP = {mAP:.4f}")
        print(f"wrote {args.out}")
    except Exception as e:
        print(f"Error evaluating: {e}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "rank1": 0.0,
                "mAP": 0.0,
                "n_query": 0,
                "n_gallery": 0,
                "no_pos": 0,
                "similarity_shape": [0, 0],
                "error": str(e)
            }, f, indent=2)
        print(f"Created placeholder results with error: {e}")

if __name__=="__main__":
    main()