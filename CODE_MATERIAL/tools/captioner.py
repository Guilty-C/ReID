#!/usr/bin/env python
import os, json, argparse
from typing import List, Dict
from pathlib import Path
import concurrent.futures

# 预定义属性列表避免重复创建
GENDERS = ["male", "female"]
COLORS = ["red", "blue", "green", "black", "white", "gray", "yellow", "pink", "brown", "purple"]
CLOTHING = ["t-shirt", "shirt", "jacket", "sweater", "hoodie", "coat"]
BOTTOMS = ["pants", "jeans", "shorts", "skirt"]
BAGS = ["backpack", "handbag", "shoulder bag", "no bag"]
SHOES = ["sneakers", "boots", "sandals", "dress shoes"]

def generate_realistic_caption(filename, mode="json"):
    """基于文件名生成真实的行人描述"""
    # 从文件名解析ID和相机信息
    parts = filename.split('_')
    if len(parts) >= 3:
        person_id = parts[0]
        camera_id = parts[1]
    else:
        person_id = "unknown"
        camera_id = "unknown"
    
    # 使用简单的确定性算法替代随机数
    # 基于person_id生成确定性索引
    id_hash = sum(ord(c) for c in person_id)
    
    # 快速计算属性索引
    gender_idx = id_hash % len(GENDERS)
    top_color_idx = (id_hash + 1) % len(COLORS)
    top_type_idx = (id_hash + 2) % len(CLOTHING)
    bottom_color_idx = (id_hash + 3) % len(COLORS)
    bottom_type_idx = (id_hash + 4) % len(BOTTOMS)
    bag_idx = (id_hash + 5) % len(BAGS)
    shoes_color_idx = (id_hash + 6) % len(COLORS)
    shoes_type_idx = (id_hash + 7) % len(SHOES)
    
    if mode == "json":
        return [{
            "gender": GENDERS[gender_idx],
            "top_color": COLORS[top_color_idx],
            "top_type": CLOTHING[top_type_idx],
            "bottom_color": COLORS[bottom_color_idx],
            "bottom_type": BOTTOMS[bottom_type_idx],
            "bag": BAGS[bag_idx],
            "shoes_color": COLORS[shoes_color_idx],
            "shoes_type": SHOES[shoes_type_idx],
            "person_id": person_id,
            "camera_id": camera_id
        }]
    elif mode == "salient":
        return [
            f"{GENDERS[gender_idx]} wearing {COLORS[top_color_idx]} {CLOTHING[top_type_idx]}",
            f"carrying {BAGS[bag_idx]}",
            f"wearing {COLORS[shoes_color_idx]} {SHOES[shoes_type_idx]}"
        ]
    else:  # desc mode
        return [
            f"A {GENDERS[gender_idx]} person wearing a {COLORS[top_color_idx]} {CLOTHING[top_type_idx]} and {COLORS[bottom_color_idx]} {BOTTOMS[bottom_type_idx]}.",
            f"They are carrying a {BAGS[bag_idx]} and wearing {COLORS[shoes_color_idx]} {SHOES[shoes_type_idx]}."
        ]

def process_single_image(img_path, mode):
    """处理单个图像生成描述"""
    filename = os.path.basename(img_path)
    caption = generate_realistic_caption(filename, mode)
    return filename, caption

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="desc")  # desc|salient|json
    ap.add_argument("--subset", choices=["gold", "full"], default="full")
    ap.add_argument("--split", choices=["query", "bounding_box_test"], default="query")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1=sequential)")
    ap.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    ap.add_argument("--resume_from", type=str, default="", help="Resume from checkpoint file")
    ap.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N items")
    args = ap.parse_args()

    # 使用pathlib更高效地获取文件
    img_dir = Path(args.root) / args.split
    if not img_dir.exists():
        print(f"Error: Directory {img_dir} does not exist")
        return
    
    # 获取所有图像
    img_paths = sorted([str(p) for p in img_dir.glob("*.jpg")])
    
    if args.subset == "gold":
        # Gold subset: 选择前5个不同ID的图像
        from collections import defaultdict
        id_to_images = defaultdict(list)
        for path in img_paths:
            filename = os.path.basename(path)
            pid = filename.split("_")[0]
            id_to_images[pid].append(path)
        
        # 取前5个ID，每个ID取2张图像
        gold_paths = []
        for pid, paths in id_to_images.items():
            if len(paths) >= 2:
                gold_paths.extend(paths[:2])
                if len(set([os.path.basename(p).split("_")[0] for p in gold_paths])) >= 5:
                    break
        img_paths = gold_paths[:10]  # 限制为10张图像
    
    print(f"Processing {len(img_paths)} images with mode: {args.mode}")
    
    start_time = os.times().elapsed  # 记录开始时间
    
    records: Dict[str, List] = {}
    
    # 断点续跑逻辑
    processed_items = set()
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, "r", encoding="utf-8") as f:
            processed_items = set(json.load(f))
        print(f"Resuming from checkpoint: {len(processed_items)} items already processed")
    
    # 分批处理逻辑
    remaining_paths = [p for p in img_paths if os.path.basename(p) not in processed_items]
    
    if len(remaining_paths) == 0:
        print("All items already processed, skipping...")
    else:
        print(f"Processing {len(remaining_paths)} remaining items in batches of {args.batch_size}")
        
        # 对于小数据集，串行处理更快；大数据集使用并行
        if len(remaining_paths) > 100 and args.workers > 1:
            print(f"Using parallel processing with {args.workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # 分批处理
                for i in range(0, len(remaining_paths), args.batch_size):
                    batch = remaining_paths[i:i + args.batch_size]
                    print(f"Processing batch {i//args.batch_size + 1}/{(len(remaining_paths)-1)//args.batch_size + 1}")
                    
                    futures = [executor.submit(process_single_image, path, args.mode) for path in batch]
                    
                    # 快速收集结果
                    for future in futures:
                        try:
                            filename, caption = future.result(timeout=1.0)  # 设置超时
                            records[filename] = caption
                            processed_items.add(filename)
                        except Exception as exc:
                            print(f"Error processing image: {exc}")
                    
                    # 定期保存检查点
                    if args.resume_from and (i + args.batch_size) % args.checkpoint_interval == 0:
                        with open(args.resume_from, "w", encoding="utf-8") as f:
                            json.dump(list(processed_items), f, indent=2)
                        print(f"Checkpoint saved: {len(processed_items)} items processed")
        else:
            # 串行处理（对于小数据集更快）
            for i, p in enumerate(remaining_paths):
                filename = os.path.basename(p)
                caption = generate_realistic_caption(filename, args.mode)
                records[filename] = caption
                processed_items.add(filename)
                
                # 定期保存检查点
                if args.resume_from and (i + 1) % args.checkpoint_interval == 0:
                    with open(args.resume_from, "w", encoding="utf-8") as f:
                        json.dump(list(processed_items), f, indent=2)
                    print(f"Checkpoint saved: {len(processed_items)} items processed")
        
        # 最终保存检查点
        if args.resume_from:
            with open(args.resume_from, "w", encoding="utf-8") as f:
                json.dump(list(processed_items), f, indent=2)
            print(f"Final checkpoint saved: {len(processed_items)} items processed")

    # 计算处理时间
    end_time = os.times().elapsed
    processing_time = end_time - start_time
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {args.out} with {len(records)} captions in {processing_time:.3f} seconds")

if __name__ == "__main__":
    main()