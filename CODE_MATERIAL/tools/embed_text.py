#!/usr/bin/env python
import json, argparse, numpy as np, os, sys
import torch
import torch.nn.functional as F
from PIL import Image
# import clip  # lazy import inside CLIP backends to avoid dependency when using mock
from typing import List
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class EmbeddingBackend:
    """Registry for text embedding backends"""
    
    backends = {}
    
    @classmethod
    def register(cls, name):
        def decorator(fn):
            cls.backends[name] = fn
            return fn
        return decorator
    
    @classmethod
    def get_backend(cls, name, **kwargs):
        if name not in cls.backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(cls.backends.keys())}")
        return cls.backends[name](**kwargs)

@EmbeddingBackend.register("mock")
def create_mock_backend(**kwargs):
    """Mock backend for testing"""
    class MockBackend:
        def embed_texts(self, texts: List[str]):
            n = len(texts)
            return np.random.randn(n, 768).astype("float32")
    return MockBackend()

@EmbeddingBackend.register("clip_b16")
def create_clip_b16_backend(weights_path=None, device="auto"):
    """CLIP ViT-B/16 backend"""
    # lazy import to avoid hard dependency when using mock backend
    import clip
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if weights_path and os.path.exists(weights_path):
        model, preprocess = clip.load("ViT-B/16", device=device, download_root=weights_path)
    else:
        model, preprocess = clip.load("ViT-B/16", device=device)
    
    class CLIPBackend:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            
        def embed_texts(self, texts: List[str]):
            with torch.no_grad():
                text_inputs = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = self.model.encode_text(text_inputs)
                text_features = F.normalize(text_features, dim=-1)
                return text_features.cpu().numpy().astype("float32")
    
    return CLIPBackend(model, device)

@EmbeddingBackend.register("clip_l14")
def create_clip_l14_backend(weights_path=None, device="auto"):
    """CLIP ViT-L/14 backend"""
    # lazy import to avoid hard dependency when using mock backend
    import clip
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if weights_path and os.path.exists(weights_path):
        model, preprocess = clip.load("ViT-L/14", device=device, download_root=weights_path)
    else:
        model, preprocess = clip.load("ViT-L/14", device=device)
    
    class CLIPBackend:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            
        def embed_texts(self, texts: List[str]):
            with torch.no_grad():
                text_inputs = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = self.model.encode_text(text_inputs)
                text_features = F.normalize(text_features, dim=-1)
                return text_features.cpu().numpy().astype("float32")
    
    return CLIPBackend(model, device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--weights-path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    
    # Load config if not provided via CLI (optional)
    config = {
        "embed": {
            "text_backend": args.backend,
            "weights_path": args.weights_path,
            "device": args.device,
            "batch_size": args.batch_size,
        }
    }
    try:
        import yaml
        with open("configs/reid.yaml", "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if isinstance(yaml_cfg, dict) and "embed" in yaml_cfg:
                for k in ["text_backend", "weights_path", "device", "batch_size"]:
                    if yaml_cfg["embed"].get(k) is not None and getattr(args, k.replace('text_', ''), None) is None:
                        config["embed"][k] = yaml_cfg["embed"][k]
    except Exception:
        pass
    
    backend_name = config["embed"]["text_backend"] or "mock"
    weights_path = config["embed"]["weights_path"]
    device = config["embed"]["device"]
    batch_size = config["embed"]["batch_size"] or 64
    
    print(f"Using text backend: {backend_name}, device: {device}, batch_size: {batch_size}")
    
    # Initialize backend
    backend = EmbeddingBackend.get_backend(
        backend_name, 
        weights_path=weights_path, 
        device=device
    )
    
    # Load captions
    with open(args.captions, "r", encoding="utf-8") as f:
        caps = json.load(f)
    
    # Prepare texts for embedding
    texts = []
    image_names = []
    # Preserve insertion order from JSON object
    for img_name, caption_data in caps.items():
        if isinstance(caption_data, list) and caption_data:
            if isinstance(caption_data[0], dict):  # JSON mode -> single aggregated string
                text_parts = []
                for attr_dict in caption_data:
                    if isinstance(attr_dict, dict):
                        text_parts.append(" ".join(f"{k}:{v}" for k, v in attr_dict.items() if v != "[TBD]"))
                group = [". ".join(text_parts) if text_parts else "person"]
            else:  # desc/salient/api mode -> list of strings; optionally split comma tokens
                # Normalize separators and split when only one long string present
                items = [str(x) for x in caption_data if x != "[TBD]"] or ["person"]
                # Detect structured three-line schema: LINE 1 (key=value; ...), LINE 2 (cues), LINE 3 (context)
                group = None
                if len(items) == 1:
                    s = items[0]
                    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
                    if len(lines) >= 2 and ("gender=" in lines[0] and "distinctive=" in lines[0]):
                        line1 = lines[0]
                        line2 = lines[1]
                        # Parse LINE 1 key=value pairs separated by ';'
                        kv = {}
                        for part in [p.strip() for p in line1.split(";") if p.strip()]:
                            if "=" in part:
                                k, v = part.split("=", 1)
                                kv[k.strip()] = v.strip()
                        def add_phrase(ph: str):
                            nonlocal group
                            if group is None:
                                group = []
                            if ph and ph.strip() and ph.strip().lower() not in ("none", "unknown"):
                                group.append(ph.strip())
                        # Compose clothing/accessory phrases from LINE 1
                        top_color = kv.get("top_color", "unknown")
                        top_type = kv.get("top_type", "unknown")
                        bottom_color = kv.get("bottom_color", "unknown")
                        bottom_type = kv.get("bottom_type", "unknown")
                        bottom_length = kv.get("bottom_length", "unknown")
                        shoes_color = kv.get("shoes_color", "unknown")
                        shoes_type = kv.get("shoes_type", "unknown")
                        bag_color = kv.get("bag_color", "unknown")
                        bag_type = kv.get("bag_type", "unknown")
                        headwear = kv.get("headwear", "unknown")
                        hair = kv.get("hair", "unknown")
                        accessories = kv.get("accessories", "none")
                        pattern = kv.get("pattern", "none")
                        distinctive = kv.get("distinctive", "none")
                        # Dress rule
                        if bottom_type == "dress":
                            if bottom_color not in ("unknown", "none"):
                                add_phrase(f"{bottom_color} dress")
                        else:
                            if top_color not in ("unknown", "none") and top_type not in ("unknown", "none"):
                                add_phrase(f"{top_color} {top_type}")
                            if bottom_color not in ("unknown", "none") and bottom_type not in ("unknown", "none"):
                                if bottom_length not in ("unknown", "none"):
                                    add_phrase(f"{bottom_length} {bottom_color} {bottom_type}")
                                else:
                                    add_phrase(f"{bottom_color} {bottom_type}")
                        if shoes_color not in ("unknown", "none") and shoes_type not in ("unknown", "none"):
                            add_phrase(f"{shoes_color} {shoes_type}")
                        if bag_type not in ("none", "unknown"):
                            if bag_color not in ("unknown", "none"):
                                add_phrase(f"{bag_color} {bag_type}")
                            else:
                                add_phrase(f"{bag_type}")
                        if headwear not in ("none", "unknown"):
                            add_phrase(headwear)
                        if hair not in ("unknown", "none"):
                            add_phrase(f"{hair} hair" if hair not in ("bald", "covered") else hair)
                        if accessories not in ("none", "unknown"):
                            add_phrase(accessories)
                        if pattern not in ("none", "unknown"):
                            add_phrase("striped" if pattern == "stripes" else pattern)
                        add_phrase(distinctive)
                        # Parse LINE 2 cues: "cue1 | cue2 | cue3"
                        cues = [c.strip() for c in line2.split("|") if c.strip()]
                        for c in cues:
                            add_phrase(c)
                    # If not schema, fall-through to legacy parsing
                if group is None:
                    # Detect structured two-line output: CAPTION + TAGS
                    tags_phrases = []
                    caption_line = ""
                    for it in items:
                        low = it.lower().strip()
                        if low.startswith("tags:"):
                            tags_str = it.split(":", 1)[1].strip()
                            raw_tokens = [t.strip() for t in tags_str.split(",") if t.strip()]
                            def tok_to_phrase(tok: str) -> str:
                                t = tok.strip().lower()
                                # normalize no-* tokens
                                if t.startswith("no-"):
                                    return t.replace("no-", "no ")
                                parts = t.split("-")
                                # handle 2-part tokens for shoes/bag
                                if parts[0] in ("shoes","bag") and len(parts) == 2:
                                    return parts[1]
                                if len(parts) >= 3 and parts[0] in ("top","outerwear","bottom","dress","shoes","bag"):
                                    cat, typ, color = parts[0], parts[1], parts[2]
                                    if cat == "dress":
                                        return f"{color} dress"
                                    return f"{color} {typ}"
                                if parts[0] in ("logo","number","text","pattern"):
                                    if len(parts) >= 2:
                                        return f"{parts[1]} {parts[0]}"
                                    return parts[0]
                                if parts[0].startswith("hair") and len(parts) >= 3:
                                    return f"{parts[1]} {parts[2]} hair"
                                return tok
                            tags_phrases = [tok_to_phrase(t) for t in raw_tokens]
                        elif low.startswith("caption:"):
                            caption_line = it.split(":", 1)[1].strip()
                    if tags_phrases:
                        clothing_kw_all = [
                            "t-shirt","shirt","jacket","sweater","hoodie","coat",
                            "blouse","top","jeans","pants","shorts","skirt","trousers","dress","tee"
                        ]
                        cloth_phrases = [p for p in tags_phrases if any(k in p.lower() for k in clothing_kw_all)]
                        group = cloth_phrases or tags_phrases
                        top_kw = ["t-shirt","shirt","jacket","sweater","hoodie","coat","blouse","top","tee"]
                        bottom_kw = ["jeans","pants","shorts","skirt","trousers"]
                        top_tok = next((p for p in tags_phrases if any(k in p.lower() for k in top_kw)), "")
                        bottom_tok = next((p for p in tags_phrases if any(k in p.lower() for k in bottom_kw)), "")
                        dress_tok = next((p for p in tags_phrases if "dress" in p.lower()), "")
                        phrase = ""
                        if top_tok and bottom_tok:
                            phrase = f"person wearing {top_tok} and {bottom_tok}"
                        elif dress_tok:
                            phrase = f"person wearing {dress_tok}"
                        elif top_tok:
                            phrase = f"person wearing {top_tok}"
                        elif bottom_tok:
                            phrase = f"person wearing {bottom_tok}"
                        if phrase:
                            group = [phrase] + group
                        if caption_line:
                            group = [caption_line] + group
                    else:
                        if len(items) == 1:
                            s = items[0].replace("；", ",").replace("，", ",")
                            tokens = [t.strip() for t in s.split(",") if t.strip()]
                            group = tokens if len(tokens) >= 2 else items
                            if len(tokens) >= 2:
                                clothing_kw_all = [
                                    "t-shirt","shirt","jacket","sweater","hoodie","coat",
                                    "blouse","top","jeans","pants","shorts","skirt","trousers","dress","tee"
                                ]
                                cloth_tokens = [t for t in tokens if any(k in t.lower() for k in clothing_kw_all)]
                                if len(cloth_tokens) >= 1:
                                    group = cloth_tokens
                                top_kw = ["t-shirt","shirt","jacket","sweater","hoodie","coat","blouse","top","tee"]
                                bottom_kw = ["jeans","pants","shorts","skirt","trousers"]
                                top_tok = next((t for t in tokens if any(k in t.lower() for k in top_kw)), "")
                                bottom_tok = next((t for t in tokens if any(k in t.lower() for k in bottom_kw)), "")
                                dress_tok = next((t for t in tokens if "dress" in t.lower()), "")
                                phrase2 = ""
                                if top_tok and bottom_tok:
                                    phrase2 = f"person wearing {top_tok} and {bottom_tok}"
                                elif dress_tok:
                                    phrase2 = f"person wearing {dress_tok}"
                                elif top_tok:
                                    phrase2 = f"person wearing {top_tok}"
                                elif bottom_tok:
                                    phrase2 = f"person wearing {bottom_tok}"
                                if phrase2:
                                    group = [phrase2] + group
                        else:
                            group = items
        else:
            group = ["person"]
        grouped_texts.append(group)
        image_names.append(img_name)
    
    # Batch processing with progress bar (auto-disables in non-TTY)
    total_batches = (len(texts) - 1) // batch_size + 1 if len(texts) > 0 else 0
    use_tqdm = (tqdm is not None) and (sys.stdout.isatty() or os.environ.get("TQDM_FORCE") == "1")
    embeddings = []
    if use_tqdm:
        bar = tqdm(total=total_batches, desc="Embedding batches", unit="batch", leave=False)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeds = backend.embed_texts(batch_texts)
        embeddings.append(batch_embeds)
        current_batch = i // batch_size + 1
        if use_tqdm:
            bar.update(1)
            bar.set_postfix({"batch": f"{current_batch}/{total_batches}", "size": len(batch_texts)})
        else:
            print(f"Embedding batch {current_batch}/{total_batches}")
    if use_tqdm:
        bar.close()
    
    # Concatenate all embeddings
    if embeddings:
        all_embeds = np.concatenate(embeddings, axis=0).astype("float32")
    else:
        all_embeds = np.zeros((0, 768), dtype="float32")
    
    # Validate embeddings
    assert not np.any(np.isnan(all_embeds)), "Embeddings contain NaN values"
    assert not np.any(np.isinf(all_embeds)), "Embeddings contain Inf values"
    assert all_embeds.shape[0] == len(texts), f"Shape mismatch: {all_embeds.shape[0]} vs {len(texts)}"
    
    print(f"Generated embeddings: {all_embeds.shape}")
    
    # Save embeddings
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, all_embeds)
    print(f"Saved text embeddings to {args.out}")

if __name__ == "__main__":
    main()


def _token_weight(token: str) -> float:
    t = token.lower()
    clothing_kw = ["t-shirt","shirt","jacket","sweater","hoodie","coat","jeans","pants","shorts","skirt","dress","blouse","tee","trousers"]
    shoes_kw = ["sneakers","boots","sandals","shoes","heels","loafers"]
    bag_kw = ["backpack","handbag","shoulder bag","bag","shoulder","waist","hand"]
    acc_kw = ["hat","glasses","sunglasses","cap","mask","scarf","watch"]
    hair_kw = ["hair","curly","straight","long","short","ponytail","bald","covered"]
    motion_kw = ["walking","standing","sitting","running","biking","cycling","phone"]
    skin_kw = ["skin"]
    if any(k in t for k in clothing_kw):
        return 1.0
    if any(k in t for k in shoes_kw):
        return 0.2
    if any(k in t for k in bag_kw):
        return 0.1
    if any(k in t for k in acc_kw):
        return 0.1
    if any(k in t for k in hair_kw):
        return 0.05
    if any(k in t for k in motion_kw):
        return 0.05
    if any(k in t for k in skin_kw):
        return 0.0
    return 0.1