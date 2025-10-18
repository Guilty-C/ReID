import os, sys, json, shutil, subprocess, time, numpy as np, pathlib

def _mk_mock_ds(root):
    q = os.path.join(root, "query"); g = os.path.join(root, "bounding_box_test")
    os.makedirs(q, exist_ok=True); os.makedirs(g, exist_ok=True)
    for d, n in [(q, 8), (g, 16)]:
        for i in range(n):
            open(os.path.join(d, f"mock_{i:03d}.jpg"), "wb").close()

def test_end2end_smoke(tmp_path):
    workdir = os.getenv("WORKDIR", ".")
    data_root = os.getenv("DATASET_ROOT", "")
    if os.getenv("CI_MOCK", "0") == "1" or not data_root or not os.path.isdir(data_root):
        data_root = str(tmp_path / "data" / "mock_market")
        _mk_mock_ds(data_root)

    outputs = os.path.join(workdir, "outputs", "ci_smoke")
    os.makedirs(outputs, exist_ok=True)

    # 1) captions
    subprocess.check_call([sys.executable, os.path.join(workdir, "tools", "captioner.py"),
                           "--root", data_root, "--out", os.path.join(outputs, "captions.json"),
                           "--mode", "json"])
    # 2) text embeds
    subprocess.check_call([sys.executable, os.path.join(workdir, "tools", "embed_text.py"),
                           "--captions", os.path.join(outputs, "captions.json"),
                           "--out", os.path.join(outputs, "text_embeds.npy")])
    # 3) image embeds
    subprocess.check_call([sys.executable, os.path.join(workdir, "tools", "embed_image.py"),
                           "--root", data_root,
                           "--out", os.path.join(outputs, "img_embeds.npy")])
    # 4) retrieve + eval
    metrics = os.path.join(outputs, "metrics.json")
    subprocess.check_call([sys.executable, os.path.join(workdir, "tools", "retrieve_eval.py"),
                           "--text", os.path.join(outputs, "text_embeds.npy"),
                           "--img", os.path.join(outputs, "img_embeds.npy"),
                           "--out", metrics])

    with open(metrics, "r") as f:
        m = json.load(f)
    for k in ["rank1","mAP","n_query","n_gallery"]:
        assert k in m and isinstance(m[k], (int, float))