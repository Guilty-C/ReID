import csv, collections, os

# 读取文件列表
def read_list(p):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    return []

# 读取manifest.csv
rows = list(csv.DictReader(open("manifest.csv", encoding="utf-8")))
ql = read_list("filelists/query.txt")
gl = read_list("filelists/gallery.txt")

# 获取ID信息
by_path = {r["image_path"]: r for r in rows}
qid = {by_path.get(p, {}).get("id", "[NA]") for p in ql if p in by_path}
gid = {by_path.get(p, {}).get("id", "[NA]") for p in gl if p in by_path}

# 计算交集
overlap = len(qid & gid)

# 输出结果
os.makedirs("outputs/diagnostics", exist_ok=True)
with open("outputs/diagnostics/id_overlap.txt", "w") as f:
    f.write(f"query_ids={len(qid)}\ngallery_ids={len(gid)}\noverlap_ids={overlap}\n")

print(f"[STATS] query_ids={len(qid)}, gallery_ids={len(gid)}, overlap_ids={overlap}")