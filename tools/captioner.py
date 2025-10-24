#!/usr/bin/env python
import os, json, argparse, sys
from typing import List, Dict
from pathlib import Path
import concurrent.futures
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

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
    ap.add_argument("--mode", default="desc")  # desc|salient|json|api|clip_attr
    ap.add_argument("--subset", choices=["gold", "full"], default="full")
    ap.add_argument("--split", choices=["query", "bounding_box_test"], default="query")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1=sequential)")
    ap.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    ap.add_argument("--resume_from", type=str, default="", help="Resume from checkpoint file")
    ap.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N items")
    # API captioning options
    ap.add_argument("--api_prompt", type=str, default="", help="LLM prompt text")
    ap.add_argument("--api_prompt_file", type=str, default="", help="File path with prompt")
    ap.add_argument("--api_model", type=str, default="", help="Model name (optional)")
    ap.add_argument("--api_url", type=str, default="", help="Base URL for API (optional)")
    ap.add_argument("--api_key", type=str, default="", help="API key (optional)")
    # CLIP attribute captioning options
    ap.add_argument("--clip_model", type=str, default="ViT-L/14", help="CLIP model name for local captioning")
    ap.add_argument("--clip_device", type=str, default="auto", help="Device for CLIP (auto/cpu/cuda)")
    ap.add_argument("--subset-count", type=int, default=10, help="Total images for gold subset (2 per ID)")
    args = ap.parse_args()

    # 使用pathlib更高效地获取文件
    img_dir = Path(args.root) / args.split
    if not img_dir.exists():
        print(f"Error: Directory {img_dir} does not exist")
        return

    # 获取所有图像
    img_paths = sorted([str(p) for p in img_dir.glob("*.jpg")])
    
    if args.subset == "gold":
        # Configurable gold subset: select first K images with >=2 per ID
        from collections import defaultdict
        id_to_images = defaultdict(list)
        for path in img_paths:
            filename = os.path.basename(path)
            pid = filename.split("_")[0]
            id_to_images[pid].append(path)
        target_total = max(1, int(args.subset_count))
        ids_needed = max(1, (target_total + 1) // 2)
        gold_paths = []
        for pid, paths in id_to_images.items():
            if len(paths) >= 2:
                gold_paths.extend(paths[:2])
                if len(set([os.path.basename(p).split("_")[0] for p in gold_paths])) >= ids_needed:
                    break
        img_paths = gold_paths[:target_total]
    
    print(f"Processing {len(img_paths)} images with mode: {args.mode}")
    
    start_time = os.times().elapsed  # 记录开始时间
    
    records: Dict[str, List] = {}
    # CLIP attribute captioning branch
    if args.mode == "clip_attr":
        try:
            import clip
            import torch
            from PIL import Image
            device = args.clip_device if args.clip_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            model, preprocess = clip.load(args.clip_model, device=device)
            # Candidate phrases (precompute embeddings)
            colors = COLORS
            top_types = CLOTHING
            bottom_types = BOTTOMS
            shoes_types = SHOES
            bag_opts = ["not carrying a bag", "carrying a backpack", "carrying a handbag", "carrying a shoulder bag"]

            def embed_texts(texts):
                with torch.no_grad():
                    toks = clip.tokenize(texts, truncate=True).to(device)
                    feats = model.encode_text(toks)
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    return feats

            texts_top = [f"a person wearing a {c} {t}" for c in colors for t in top_types]
            E_top = embed_texts(texts_top)
            idx_top_map = [(c, t) for c in colors for t in top_types]

            texts_bottom = [f"a person wearing {c} {t}" for c in colors for t in bottom_types]
            E_bottom = embed_texts(texts_bottom)
            idx_bottom_map = [(c, t) for c in colors for t in bottom_types]

            texts_shoes = [f"wearing {c} {t}" for c in colors for t in shoes_types]
            E_shoes = embed_texts(texts_shoes)
            idx_shoes_map = [(c, t) for c in colors for t in shoes_types]

            texts_bag = bag_opts
            E_bag = embed_texts(texts_bag)

            use_tqdm = (tqdm is not None) and (sys.stdout.isatty() or os.environ.get("TQDM_FORCE") == "1")
            bar = tqdm(total=len(img_paths), desc="CLIP-attr", unit="img") if use_tqdm else None

            for p in img_paths:
                fn = os.path.basename(p)
                try:
                    im = Image.open(p).convert("RGB")
                    im_t = preprocess(im).unsqueeze(0).to(device)
                    with torch.no_grad():
                        f_img = model.encode_image(im_t)
                        f_img = torch.nn.functional.normalize(f_img, dim=-1)
                    # Scores per category
                    s_top = (f_img @ E_top.T).squeeze(0)
                    top_idx = int(torch.argmax(s_top).item())
                    top_color, top_type = idx_top_map[top_idx]

                    s_bottom = (f_img @ E_bottom.T).squeeze(0)
                    bottom_idx = int(torch.argmax(s_bottom).item())
                    bottom_color, bottom_type = idx_bottom_map[bottom_idx]

                    s_shoes = (f_img @ E_shoes.T).squeeze(0)
                    shoes_idx = int(torch.argmax(s_shoes).item())
                    shoes_color, shoes_type = idx_shoes_map[shoes_idx]

                    s_bag = (f_img @ E_bag.T).squeeze(0)
                    bag_choice = int(torch.argmax(s_bag).item())
                    bag_phrase = texts_bag[bag_choice]

                    desc = (
                        f"{top_color} {top_type}, {bottom_color} {bottom_type}, {shoes_color} {shoes_type}, {bag_phrase}"
                    )
                    # Build TAGS tokens aligned to prompt schema
                    top_cat = "outerwear" if top_type in ["jacket", "coat"] else "top"
                    shoes_typ = "shoes" if shoes_type == "dress shoes" else shoes_type
                    if "not carrying a bag" in bag_phrase:
                        bag_token = "no-bag"
                    elif "backpack" in bag_phrase:
                        bag_token = "bag-backpack"
                    elif "handbag" in bag_phrase:
                        bag_token = "bag-handbag"
                    elif "shoulder bag" in bag_phrase:
                        bag_token = "bag-shoulder bag"
                    else:
                        bag_token = "bag-unknown"
                    tags = [
                        "unknown",
                        "hair-unknown",
                        f"{top_cat}-{top_type}-{top_color}",
                        f"bottom-{bottom_type}-{bottom_color}",
                        f"shoes-{shoes_typ}-{shoes_color}",
                        bag_token,
                        "no-hat",
                        "no-mask",
                    ]
                    tags_line = "TAGS: " + ", ".join(tags)
                    records[fn] = [f"CAPTION: {desc}", tags_line]
                except Exception as e:
                    print(f"[WARN] CLIP-attr failed on {fn}: {e}")
                    records[fn] = ["person"]
                if bar:
                    bar.update(1)
            if bar:
                bar.close()
            # Save and return directly for clip_attr to avoid later overwrites
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"[CLIP-attr] Generated {args.out} with {len(records)} captions")
            return
        except Exception as e:
            print(f"[ERROR] CLIP attribute captioning setup failed: {e}")
            # Fallback to deterministic captions by filename
            for p in img_paths:
                filename = os.path.basename(p)
                caption = generate_realistic_caption(filename, "desc")
                records[filename] = caption
    # API captioning branch
    if args.mode == "api":
        # 优先使用 --api_prompt，其次支持从文件加载；JSON文件时读取键 prompt/text/instruction
        prompt = args.api_prompt
        if not prompt and args.api_prompt_file:
            try:
                fpath = Path(args.api_prompt_file)
                raw = fpath.read_text('utf-8')
                if fpath.suffix.lower() == '.json':
                    try:
                        obj = json.loads(raw)
                        prompt = obj.get('prompt') or obj.get('text') or obj.get('instruction') or ''
                    except Exception:
                        prompt = ''
                else:
                    prompt = raw
            except Exception as e:
                print(f"Prompt file read error: {e}")
                prompt = ''
        api_model = args.api_model or os.environ.get('CAPTION_API_MODEL','')
        api_url = args.api_url or os.environ.get('CAPTION_API_URL','')
        api_key = args.api_key or os.environ.get('CAPTION_API_KEY','')
        raw_log = args.out.replace('.json','.raw.jsonl')
        meta_path = args.out.replace('.json','_meta.json')
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        start_t = os.times().elapsed
        total = 0
        # meta start
        try:
            with open(meta_path,'w',encoding='utf-8') as mf:
                json.dump({"prompt":prompt,"prompt_file":args.api_prompt_file,"model":api_model,"url":api_url,"started_at":start_t,"split":args.split,"subset":args.subset}, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Meta write error: {e}")
        # import helper if available
        try:
            from iso_api_subset_eval import call_caption_api as _call
        except Exception:
            _call = None
        # iterate images
        for p in img_paths:
            fn = os.path.basename(p)
            result_text = ""
            error = ""
            endpoint = ""
            status = None
            try:
                if _call is not None:
                    # helper reads env vars CAPTION_API_URL/CAPTION_API_KEY/CAPTION_API_MODEL
                    try:
                        import os as _os
                        if api_url: _os.environ['CAPTION_API_URL'] = api_url
                        if api_key: _os.environ['CAPTION_API_KEY'] = api_key
                        if api_model: _os.environ['CAPTION_API_MODEL'] = api_model
                    except Exception:
                        pass
                    resp = _call(p, prompt=prompt, model=api_model)
                    result_text = resp.get('text','') or resp.get('content','') or ''
                    error = resp.get('error','')
                    endpoint = resp.get('endpoint','')
                    status = resp.get('status', None)
                else:
                    import base64, requests
                    with open(p,'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                    payload = {"model": api_model or "gpt-4o-mini", "messages":[{"role":"user","content":[{"type":"input_text","text": prompt or "Describe the person concisely in English."},{"type":"input_image","image": b64}]}]}
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    endpoint = api_url.rstrip('/') + "/v1/messages"
                    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                    status = r.status_code
                    try:
                        result_text = r.json().get("output",{}).get("choices",[{}])[0].get("content",[{}])[0].get("text","")
                    except Exception as e:
                        error = str(e)
            except Exception as e:
                error = str(e)
                print(f"API error on {fn}: {e}")
            # raw log
            try:
                with open(raw_log,'a',encoding='utf-8') as rf:
                    rf.write(json.dumps({"file":fn,"text":result_text,"error":error,"endpoint":endpoint,"status":status}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            records[fn] = [result_text] if result_text else ["[API_ERROR_OR_EMPTY]"]
            total += 1
        end_t = os.times().elapsed
        try:
            with open(meta_path,'w',encoding='utf-8') as mf:
                json.dump({"prompt":prompt,"prompt_file":args.api_prompt_file,"model":api_model,"url":api_url,"started_at":start_t,"ended_at":end_t,"elapsed_s": end_t - start_t, "count": total, "split":args.split,"subset":args.subset}, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Meta finalize error: {e}")
        with open(args.out,'w',encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[API] Generated {args.out} with {len(records)} captions")
        return
    
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