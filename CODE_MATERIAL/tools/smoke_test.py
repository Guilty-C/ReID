#!/usr/bin/env python
import os, sys, json, argparse, subprocess, tempfile, time, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/reid.yaml")
    ap.add_argument("--dataset-root", default=os.getenv("DATASET_ROOT", ""))
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    workdir = os.getenv("WORKDIR", ".")
    ds = args.dataset_root
    if os.getenv("CI_MOCK","0")=="1" or not ds or not os.path.isdir(ds):
        ds = os.path.join(workdir, "data", "mock_market")
        q = os.path.join(ds,"query"); g=os.path.join(ds,"bounding_box_test")
        os.makedirs(q, exist_ok=True); os.makedirs(g, exist_ok=True)
        for d, n in [(q, 8),(g,16)]:
            for i in range(n): open(os.path.join(d,f"mock_{i:03d}.jpg"),"wb").close()

    out = os.path.join(workdir, args.out, "smoke")
    os.makedirs(out, exist_ok=True)

    cmds = [
        [sys.executable, "tools/captioner.py", "--root", ds, "--out", f"{out}/captions.json", "--mode", "json"],
        [sys.executable, "tools/embed_text.py", "--captions", f"{out}/captions.json", "--out", f"{out}/text_embeds.npy"],
        [sys.executable, "tools/embed_image.py", "--root", ds, "--out", f"{out}/img_embeds.npy"],
        [sys.executable, "tools/retrieve_eval.py", "--text", f"{out}/text_embeds.npy", "--img", f"{out}/img_embeds.npy", "--out", f"{out}/metrics.json"],
    ]
    for c in cmds: subprocess.check_call(c)

    with open(f"{out}/metrics.json") as f: m=json.load(f)
    print(f"[SMOKE] rank1={m.get('rank1')} mAP={m.get('mAP')} n_query={m.get('n_query')} n_gallery={m.get('n_gallery')}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())