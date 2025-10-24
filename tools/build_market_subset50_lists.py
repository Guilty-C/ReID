#!/usr/bin/env python
import os
from pathlib import Path

def main():
    root = Path('data/archive/Market-1501-v15.09.15')
    q_dir = root / 'query'
    g_dir = root / 'bounding_box_test'
    assert q_dir.is_dir(), f"Missing query dir: {q_dir}"
    assert g_dir.is_dir(), f"Missing gallery dir: {g_dir}"
    # index gallery by ID
    g_map = {}
    for p in sorted(g_dir.glob('*.jpg')):
        pid = p.name.split('_')[0]
        g_map.setdefault(pid, []).append(str(p.resolve()))
    # select 50 IDs present in both
    ids = []
    q_paths = []
    g_paths = []
    for p in sorted(q_dir.glob('*.jpg')):
        pid = p.name.split('_')[0]
        if pid in g_map and pid not in ids:
            ids.append(pid)
            q_paths.append(str(p.resolve()))
            g_paths.append(g_map[pid][0])
            if len(ids) >= 50:
                break
    assert len(ids) == 50, f"Only found {len(ids)} IDs with gallery matches"
    out_q = Path('larger_iso/64/query.txt')
    out_g = Path('larger_iso/64/gallery.txt')
    out_q.parent.mkdir(parents=True, exist_ok=True)
    out_g.parent.mkdir(parents=True, exist_ok=True)
    with out_q.open('w', encoding='utf-8') as f:
        for p in q_paths:
            f.write(p + '\n')
    with out_g.open('w', encoding='utf-8') as f:
        for p in g_paths:
            f.write(p + '\n')
    print(f"Wrote {out_q} and {out_g} with 50 items each")

if __name__ == '__main__':
    main()