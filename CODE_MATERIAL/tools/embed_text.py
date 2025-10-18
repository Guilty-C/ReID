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
            if isinstance(caption_data[0], dict):  # JSON mode
                text_parts = []
                for attr_dict in caption_data:
                    if isinstance(attr_dict, dict):
                        text_parts.append(" ".join(f"{k}:{v}" for k, v in attr_dict.items() if v != "[TBD]"))
                text = ". ".join(text_parts) if text_parts else "person"
            else:  # desc/salient mode
                text = ". ".join(str(x) for x in caption_data if x != "[TBD]")
        else:
            text = "person"
        texts.append(text)
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