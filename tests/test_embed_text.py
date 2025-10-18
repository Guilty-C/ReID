import os, sys, json, tempfile, subprocess, numpy as np, pathlib

def _ensure_captions(workdir):
    outp = pathlib.Path(workdir) / "outputs" / "captions.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        return str(outp)
    # 最小占位 captions
    caps = {f"q_{i}.jpg": ["person with [TBD] jacket", "salient: [TBD]"]
            for i in range(5)}
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(caps, f, ensure_ascii=False, indent=2)
    return str(outp)

def test_embed_text_basic():
    workdir = os.getenv("WORKDIR", ".")
    captions = _ensure_captions(workdir)
    out_npy = os.path.join(workdir, "outputs", "embeds", "test_text_embeds.npy")
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    cmd = [sys.executable, os.path.join(workdir, "tools", "embed_text.py"),
           "--captions", captions, "--out", out_npy]
    subprocess.check_call(cmd)
    arr = np.load(out_npy)
    assert arr.ndim == 2 and arr.shape[0] > 0
    assert str(arr.dtype).startswith("float")
    assert np.isfinite(arr).all()