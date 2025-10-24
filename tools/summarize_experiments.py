import os
import json
import csv
import sys
import datetime
from typing import List, Dict

DATE_FMT = "%Y%m%d"
ROOT_EXP = os.path.join("outputs", "experiments")

CATEGORIES = [
    ("three_line", "三行结构"),
    ("two_line", "两行结构"),
    ("color_first", "颜色优先"),
    ("structured", "结构化描述"),
    ("tags_only", "仅标签"),
    ("minimal", "最小化Tokens"),
    ("tokens", "Tokens提示"),
]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_prompt_file(summary: Dict) -> str:
    cmd = summary.get("commands", {}).get("caption", [])
    for i in range(len(cmd)):
        if cmd[i] in ("--api_prompt_file", "--prompt-file", "--prompt_file"):
            if i + 1 < len(cmd):
                return cmd[i + 1]
    return ""


def _categorize_prompt(prompt_name: str) -> str:
    name = (prompt_name or "").lower()
    for key, label in CATEGORIES:
        if key in name:
            return label
    return "通用描述"


def _is_error_run(exp_dir: str) -> bool:
    raw_path = os.path.join(exp_dir, "captions", "captions_query.raw.jsonl")
    if not os.path.exists(raw_path):
        return True
    try:
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                try:
                    j = json.loads(line)
                except Exception:
                    return True
                text = j.get("text", "") or ""
                err = j.get("error", "") or ""
                endpoint = j.get("endpoint", "") or ""
                if err or (text.startswith("[API_ERROR]")) or (not endpoint):
                    return True
                if i >= 10:
                    break
    except Exception:
        return True
    return False


def _is_smoke_run(summary: Dict) -> bool:
    subset = summary.get("subset", "")
    n_query = summary.get("metrics", {}).get("n_query", 0)
    dataset_root = summary.get("dataset_root", "")
    # 简单启发式：非gold、mock数据集、或查询量过小，视为smoke
    if subset != "gold":
        return True
    if "mock" in dataset_root.lower():
        return True
    if n_query < 32:
        return True
    return False


def _collect_rows(date_str: str) -> List[Dict]:
    rows = []
    if not os.path.isdir(ROOT_EXP):
        return rows
    for name in os.listdir(ROOT_EXP):
        if not name.startswith(f"EXP_{date_str}-"):
            continue
        exp_dir = os.path.join(ROOT_EXP, name)
        summary_path = os.path.join(exp_dir, "run_summary.json")
        metrics_path = os.path.join(exp_dir, "metrics", "metrics.json")
        if not (os.path.exists(summary_path) and os.path.exists(metrics_path)):
            continue
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue
        # 过滤smoke
        if _is_smoke_run(summary):
            continue
        # 过滤API错误等异常运行
        if _is_error_run(exp_dir):
            continue
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue
        prompt_file = _get_prompt_file(summary)
        prompt_name = os.path.basename(prompt_file) if prompt_file else ""
        direction = _categorize_prompt(prompt_name)
        elapsed = summary.get("elapsed_s", {})
        api = summary.get("api", {})
        rows.append({
            "exp_id": name,
            "subset": summary.get("subset", ""),
            "prompt_name": prompt_name,
            "direction": direction,
            "caption_mode": summary.get("caption", {}).get("mode", ""),
            "api_url": api.get("url", ""),
            "api_model": api.get("model", ""),
            "rank1": summary.get("metrics", {}).get("rank1", None),
            "mAP": summary.get("metrics", {}).get("mAP", None),
            "gate": summary.get("gate", ""),
            "n_query": summary.get("metrics", {}).get("n_query", 0),
            "n_gallery": summary.get("metrics", {}).get("n_gallery", 0),
            "caption_elapsed_s": elapsed.get("caption", None),
            "text_embed_elapsed_s": elapsed.get("text_embed", None),
            "image_embed_elapsed_s": elapsed.get("image_embed", None),
            "eval_elapsed_s": elapsed.get("eval", None),
        })
    return rows


def main(date_str: str) -> int:
    outdir = os.path.join("outputs", "reports", f"EXP_{date_str}")
    os.makedirs(outdir, exist_ok=True)
    rows = _collect_rows(date_str)
    csv_path = os.path.join(outdir, "summary.csv")
    json_path = os.path.join(outdir, "summary.json")

    if not rows:
        # 写入空结构，避免后续脚本失败
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["exp_id", "prompt_name", "direction", "rank1", "mAP", "gate"])
            writer.writeheader()
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"date": date_str, "rows": []}, jf, ensure_ascii=False, indent=2)
        print(f"No rows found for date {date_str}. Saved empty {csv_path} and {json_path}.")
        return 0

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"date": date_str, "rows": rows}, jf, ensure_ascii=False, indent=2)

    print(f"Saved {csv_path} ({len(rows)} rows)")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime(DATE_FMT)
    sys.exit(main(date_arg))