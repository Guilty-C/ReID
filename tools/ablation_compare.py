import os, time, json, numpy as np, csv, torch
from pathlib import Path

def main():
    workdir = os.getcwd()
    emb_dir = os.path.join(workdir, 'outputs', 'embeds')
    metrics_dir = os.path.join(workdir, 'outputs', 'metrics')
    diag_dir = os.path.join(workdir, 'outputs', 'diagnostics')
    abl_dir = os.path.join(workdir, 'outputs', 'ablation')
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(abl_dir, exist_ok=True)

    # Paths
    text_real = os.path.join(emb_dir, 'text_iso_real.npy')
    img_real = os.path.join(emb_dir, 'img_iso_real.npy')
    metrics_real = os.path.join(metrics_dir, 'metrics_iso_api_real.json')
    text_mock = os.path.join(emb_dir, 'text_iso_api.npy')
    img_mock = os.path.join(emb_dir, 'img_iso.npy')
    metrics_mock = os.path.join(metrics_dir, 'metrics_iso_api.json')

    # Measure eval elapsed for real
    start = time.time()
    os.system(f"python tools/retrieve_eval.py --text \"{text_real}\" --img \"{img_real}\" --out \"{metrics_real}\"")
    elapsed_real = time.time() - start

    # Measure eval elapsed for mock
    start = time.time()
    os.system(f"python tools/retrieve_eval.py --text \"{text_mock}\" --img \"{img_mock}\" --out \"{metrics_mock}\"")
    elapsed_mock = time.time() - start

    # Load metrics
    with open(metrics_real, 'r', encoding='utf-8') as f:
        m_real = json.load(f)
    with open(metrics_mock, 'r', encoding='utf-8') as f:
        m_mock = json.load(f)

    # iso_compare.csv
    csv_path = os.path.join(abl_dir, 'iso_compare.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant','rank1','mAP','n_query','n_gallery','elapsed_s'])
        w.writerow(['mock', m_mock.get('rank1'), m_mock.get('mAP'), m_mock.get('n_query'), m_mock.get('n_gallery'), round(elapsed_mock, 4)])
        w.writerow(['real', m_real.get('rank1'), m_real.get('mAP'), m_real.get('n_query'), m_real.get('n_gallery'), round(elapsed_real, 4)])

    # Diagnostics: top1 samples and similarity histogram (real)
    T = np.load(text_real)
    I = np.load(img_real)
    # normalize
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-8)
    S = T @ I.T

    # top1 indices for each query
    top1_idx = np.argmax(S, axis=1)
    top1_scores = S[np.arange(S.shape[0]), top1_idx]

    # Map filenames via iso lists
    iso_q = [line.strip() for line in open(os.path.join(workdir,'iso','query.txt'), encoding='utf-8') if line.strip()]
    iso_g = [line.strip() for line in open(os.path.join(workdir,'iso','gallery.txt'), encoding='utf-8') if line.strip()]
    # write top1 samples
    top1_csv = os.path.join(diag_dir, 'top1_samples_real.csv')
    with open(top1_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['q_index','q_path','g_index','g_path','similarity'])
        for i in range(S.shape[0]):
            q_path = iso_q[i] if i < len(iso_q) else ''
            gi = int(top1_idx[i])
            g_path = iso_g[gi] if gi < len(iso_g) else ''
            w.writerow([i, q_path, gi, g_path, float(top1_scores[i])])

    # similarity histogram
    hist_csv = os.path.join(diag_dir, 'sim_hist_real.csv')
    vals = S.flatten()
    counts, bins = np.histogram(vals, bins=20, range=(-1.0, 1.0))
    centers = (bins[:-1] + bins[1:]) / 2.0
    with open(hist_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['bin_center','count'])
        for c, ctr in zip(counts.tolist(), centers.tolist()):
            w.writerow([ctr, c])

    # iso_summary_real.txt
    sum_path = os.path.join(diag_dir, 'iso_summary_real.txt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write(f"n_query={m_real.get('n_query')}\n")
        f.write(f"n_gallery={m_real.get('n_gallery')}\n")
        f.write(f"rank1={m_real.get('rank1'):.4f}\n")
        f.write(f"mAP={m_real.get('mAP'):.4f}\n")
        f.write(f"device={device}\n")
        f.write(f"elapsed_eval_s={elapsed_real:.4f}\n")
        f.write("backend=clip_l14\n")
    print('[OK] compare and diagnostics generated')

if __name__ == '__main__':
    main()