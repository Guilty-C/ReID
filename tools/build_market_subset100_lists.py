#!/usr/bin/env python
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # repo root

def pick_root():
    real = BASE/'data/archive/Market-1501-v15.09.15'
    mock = BASE/'data/mock_market'
    if real.is_dir(): return real, real/'query', real/'bounding_box_test', 'real'
    else: return mock, mock/'query', mock/'bounding_box_test', 'mock'

def main():
    root, q_dir, g_dir, mode = pick_root()
    assert q_dir.is_dir(), f"Missing query dir: {q_dir}"
    assert g_dir.is_dir(), f"Missing gallery dir: {g_dir}"
    # index gallery by ID
    g_map = {}
    for p in sorted(g_dir.glob('*.jpg')):
        pid = p.name.split('_')[0]
        g_map.setdefault(pid, []).append(str(p.resolve()))
    # select up to 100 IDs present in both
    ids = []
    q_paths = []
    g_paths = []
    for p in sorted(q_dir.glob('*.jpg')):
        pid = p.name.split('_')[0]
        if pid in g_map and pid not in ids:
            ids.append(pid)
            q_paths.append(str(p.resolve()))
            g_paths.append(g_map[pid][0])
            if len(ids) >= 100:
                break
    if len(ids) < 100 and mode=='real':
        raise AssertionError(f"Only found {len(ids)} IDs with gallery matches (need 100) in {root}")
    out_q = BASE/'larger_iso/64/query100.txt'
    out_g = BASE/'larger_iso/64/gallery100.txt'
    out_q.parent.mkdir(parents=True, exist_ok=True)
    out_g.parent.mkdir(parents=True, exist_ok=True)
    with out_q.open('w', encoding='utf-8') as f:
        for p in q_paths:
            f.write(p + '\n')
    with out_g.open('w', encoding='utf-8') as f:
        for p in g_paths:
            f.write(p + '\n')
    print(f"[{mode}] Wrote {out_q} and {out_g} with {len(ids)} items each")

if __name__ == '__main__':
    main()