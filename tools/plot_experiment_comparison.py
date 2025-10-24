import os
import sys
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Map Chinese category labels to English for plotting
CH2EN = {
    '三行结构': 'Three-line schema',
    '两行结构': 'Two-line schema',
    '颜色优先': 'Color-first',
    '结构化描述': 'Structured description',
    '仅标签': 'Tags-only',
    '最小化Tokens': 'Tokens-minimal',
    'Tokens提示': 'Tokens prompt',
    '通用描述': 'General',
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r['rank1'] = float(r.get('rank1', 0) or 0)
                r['mAP'] = float(r.get('mAP', 0) or 0)
            except Exception:
                r['rank1'] = 0.0
                r['mAP'] = 0.0
            rows.append(r)
    return rows


def aggregate_by_direction(rows):
    agg = defaultdict(lambda: {'rank1_sum': 0.0, 'mAP_sum': 0.0, 'count': 0})
    for r in rows:
        direction = r.get('direction', 'General')
        agg[direction]['rank1_sum'] += r['rank1']
        agg[direction]['mAP_sum'] += r['mAP']
        agg[direction]['count'] += 1
    # compute averages
    directions = []
    rank1_avg = []
    mAP_avg = []
    counts = []
    for d, v in agg.items():
        c = max(v['count'], 1)
        directions.append(d)
        rank1_avg.append(v['rank1_sum'] / c)
        mAP_avg.append(v['mAP_sum'] / c)
        counts.append(c)
    # sort by mAP
    order = sorted(range(len(directions)), key=lambda i: mAP_avg[i])
    directions = [directions[i] for i in order]
    rank1_avg = [rank1_avg[i] for i in order]
    mAP_avg = [mAP_avg[i] for i in order]
    counts = [counts[i] for i in order]
    return directions, rank1_avg, mAP_avg, counts


def to_english(directions):
    return [CH2EN.get(d, d) for d in directions]


def plot(date_str, indir):
    csv_path = os.path.join(indir, 'summary.csv')
    if not os.path.exists(csv_path):
        print(f'CSV not found: {csv_path}')
        return 1
    rows = load_rows(csv_path)
    if not rows:
        print('No rows to plot.')
        return 0
    directions, rank1_avg, mAP_avg, counts = aggregate_by_direction(rows)
    directions_en = to_english(directions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Rank-1 panel
    axes[0].barh(directions_en, rank1_avg, color='#4C78A8')
    axes[0].set_title('Rank-1 (avg)')
    axes[0].set_xlabel('Rank-1')
    for i, v in enumerate(rank1_avg):
        axes[0].text(v, i, f'{v:.3f} / n={counts[i]}', va='center', ha='left', fontsize=9)

    # mAP panel
    axes[1].barh(directions_en, mAP_avg, color='#F58518')
    axes[1].set_title('mAP (avg)')
    axes[1].set_xlabel('mAP')
    for i, v in enumerate(mAP_avg):
        axes[1].text(v, i, f'{v:.3f} / n={counts[i]}', va='center', ha='left', fontsize=9)

    fig.suptitle(f'Prompt Direction Comparison ({date_str})', fontsize=14)
    fig.tight_layout()
    outpng = os.path.join(indir, 'comparison.png')
    fig.savefig(outpng, dpi=150)
    print(f'Saved {outpng}')
    return 0


if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else 'YYYYMMDD'
    indir = os.path.join('outputs', 'reports', f'EXP_{date_str}')
    sys.exit(plot(date_str, indir))