import os, json, time, csv
from datetime import datetime

def main():
    workdir = os.getcwd()
    metrics_dir = os.path.join(workdir, 'outputs', 'metrics')
    abl_path = os.path.join(workdir, 'outputs', 'ablation', 'iso_compare.csv')
    hist_path = os.path.join(metrics_dir, 'history.csv')
    # Read compare to get real row and elapsed
    rank1_real = mAP_real = n_q = n_g = None
    elapsed_real = None
    with open(abl_path, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    for r in rows[1:]:
        if r[0] == 'real':
            rank1_real = float(r[1])
            mAP_real = float(r[2])
            n_q = int(float(r[3]))
            n_g = int(float(r[4]))
            elapsed_real = float(r[5])
            break
    # Append to history
    now = datetime.utcnow().isoformat(timespec='seconds')
    run_id = f"iso_real_{now.replace(':','').replace('-','')}"
    dataset = f"ISO-subset-{n_q}x{n_g}"
    notes = "ISO baseline (real encoder clip_l14)"
    # Ensure file exists with header
    if not os.path.exists(hist_path):
        with open(hist_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','run_id','rank1','mAP','n_query','n_gallery','dataset','notes'])
    with open(hist_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([now, run_id, rank1_real, mAP_real, n_q, n_g, dataset, notes])
    print('[OK] appended ISO baseline to history.csv')

if __name__ == '__main__':
    main()