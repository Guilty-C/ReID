import os
import csv
import collections
import numpy as np
import shutil
from pathlib import Path

# 步骤1：创建必要的目录
os.makedirs("filelists", exist_ok=True)
os.makedirs("staging/query", exist_ok=True)
os.makedirs("staging/gallery", exist_ok=True)
os.makedirs("outputs/embeds", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# 步骤2：构建query.txt和gallery.txt文件列表
rows = list(csv.DictReader(open("manifest.csv", encoding="utf-8")))
have_split = any(r.get("split", "").lower() in ("query", "gallery") for r in rows)
by_id = collections.defaultdict(list)

for r in rows:
    by_id[r.get("id", "[TBD]")].append(r["image_path"])

q, g = [], []
if have_split:
    for r in rows:
        p = r["image_path"]
        sp = r.get("split", "").lower()
        if sp == "query":
            q.append(p)
        elif sp == "gallery":
            g.append(p)
        else:
            g.append(p)
else:
    for _id, paths in by_id.items():
        paths = sorted(paths)
        if paths:
            q.append(paths[0])
            g.extend(paths[1:])

with open("filelists/query.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(q))
with open("filelists/gallery.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(g))

print(f"文件列表构建完成: query={len(q)}, gallery={len(g)}")

# 步骤3：创建有序的staging目录
for i, path in enumerate(q, 1):
    src = path
    if not os.path.isabs(src):
        src = os.path.join(".", src)
    dst = os.path.join("staging/query", f"{i:06d}.jpg")
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"警告: 找不到源文件 {src}")
    except Exception as e:
        print(f"拷贝文件出错 {src} -> {dst}: {e}")

for i, path in enumerate(g, 1):
    src = path
    if not os.path.isabs(src):
        src = os.path.join(".", src)
    dst = os.path.join("staging/gallery", f"{i:06d}.jpg")
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"警告: 找不到源文件 {src}")
    except Exception as e:
        print(f"拷贝文件出错 {src} -> {dst}: {e}")

print("有序staging目录创建完成")

# 步骤4-7：运行嵌入和评测命令
commands = [
    "python tools/captioner.py --root staging/query --out outputs/captions_query.json",
    "python tools/embed_text.py --captions outputs/captions_query.json --out outputs/embeds/text_q.npy",
    "python tools/embed_image.py --root staging/gallery --out outputs/embeds/img_g.npy"
]

for cmd in commands:
    print(f"执行命令: {cmd}")
    os.system(cmd)

# 步骤8：归一化嵌入向量
if os.path.exists("outputs/embeds/text_q.npy") and os.path.exists("outputs/embeds/img_g.npy"):
    T = np.load("outputs/embeds/text_q.npy")
    I = np.load("outputs/embeds/img_g.npy")
    
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-9)
    
    np.save("outputs/embeds/text_q.npy", T)
    np.save("outputs/embeds/img_g.npy", I)
    
    print(f"嵌入向量归一化完成: text={T.shape}, img={I.shape}")
    
    # 步骤9：运行评测
    cmd = "python tools/retrieve_eval.py --text outputs/embeds/text_q.npy --img outputs/embeds/img_g.npy --out outputs/metrics/metrics.json"
    print(f"执行命令: {cmd}")
    os.system(cmd)
    
    # 显示评测结果
    if os.path.exists("outputs/metrics/metrics.json"):
        import json
        with open("outputs/metrics/metrics.json", "r") as f:
            metrics = json.load(f)
        print("\n评测结果:")
        print(f"Rank-1: {metrics.get('rank1', 'N/A')}")
        print(f"mAP: {metrics.get('mAP', 'N/A')}")
        print(f"n_query: {metrics.get('n_query', 'N/A')}")
        print(f"n_gallery: {metrics.get('n_gallery', 'N/A')}")
else:
    print("错误: 嵌入文件不存在，无法进行评测")