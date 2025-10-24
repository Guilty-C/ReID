#!/usr/bin/env python
import argparse, glob, os, numpy as np, sys, time, psutil
import torch
import torch.nn.functional as F
import torch.quantization
from PIL import Image
# import clip  # lazy import inside CLIP backends to avoid dependency when using mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import mmap
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import gc

class ImageEmbeddingBackend:
    """Registry for image embedding backends"""
    
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

@ImageEmbeddingBackend.register("mock")
def create_mock_image_backend(**kwargs):
    """Mock backend for testing"""
    class MockBackend:
        def embed_images(self, image_paths):
            n = len(image_paths)
            embeddings = np.random.randn(n, 768).astype("float32")
            return embeddings, image_paths  # 返回元组
    return MockBackend()

@ImageEmbeddingBackend.register("clip_b16")
def create_clip_b16_image_backend(weights_path=None, device="auto", quantize=True, low_res=True, num_processes=4, 
                                 batch_size=None, mixed_precision=False, pin_memory=True):
    """CLIP ViT-B/16 backend for images with GPU acceleration and optimization options"""
    # lazy import to avoid hard dependency when using mock backend
    import clip
    # 增强的设备检测与自动切换
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # 打印设备信息
    if device == "cpu":
        print(f"Using CPU for inference")
    else:
        print(f"Using GPU for inference: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    if weights_path and os.path.exists(weights_path):
        model, preprocess = clip.load("ViT-B/16", device=device, download_root=weights_path)
    else:
        model, preprocess = clip.load("ViT-B/16", device=device)
    
    # 根据设备类型应用不同优化
    if device == "cpu" and quantize:
        print("Applying dynamic quantization to CLIP model...")
        try:
            # 使用更安全的量化方法，避免嵌入层量化错误
            from torch.quantization import quantize_dynamic
            # 只量化线性层，避免嵌入层量化问题
            model = quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Model quantized successfully")
        except Exception as e:
            print(f"Quantization failed: {e}. Continuing without quantization.")
            # 如果量化失败，继续使用原始模型
    elif device.startswith("cuda"):
        # GPU优化：清理显存并预热模型
        torch.cuda.empty_cache()
        # 创建一个小批量数据预热模型
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input)
        print("GPU model warmed up successfully")
        
        # 添加显存优化：启用梯度检查点以减少显存使用
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing to reduce VRAM usage")
            
        # 添加显存优化：使用CUDA流水线并行化
        if torch.cuda.device_count() > 1:
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            print(f"Enabled DataParallel across {torch.cuda.device_count()} GPUs")
    
    # 创建优化的预处理函数
    if low_res:
        # 降低分辨率的预处理函数 - 使用224x224而不是112x112以匹配CLIP模型
        from torchvision import transforms
        optimized_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 保持224x224分辨率以匹配CLIP模型
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        optimized_preprocess = preprocess

    class CLIPImageBackend:
        def __init__(self, model, preprocess, optimized_preprocess, device, num_processes, 
                     batch_size=None, mixed_precision=False, pin_memory=True):
            self.model = model
            self.preprocess = preprocess
            self.optimized_preprocess = optimized_preprocess
            self.device = device
            self.num_processes = num_processes
            self.cache = {}  # 简单的缓存机制
            
            # GPU优化参数
            self.is_gpu = device.startswith("cuda")
            # GPU模式下使用更大的批处理大小
            self.batch_size = batch_size or (256 if self.is_gpu else 16)
            self.mixed_precision = mixed_precision and self.is_gpu  # 仅在GPU模式下启用混合精度
            self.pin_memory = pin_memory and self.is_gpu  # 仅在GPU模式下启用内存锁定
            
            # 如果使用GPU，创建CUDA流和混合精度scaler
            if self.is_gpu:
                # 创建多个CUDA流用于并行处理
                self.streams = [torch.cuda.Stream() for _ in range(2)]  # 两个流用于流水线处理
                self.current_stream = 0
                
                if self.mixed_precision:
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
                    print(f"Mixed precision enabled with batch size {self.batch_size}")
                else:
                    print(f"Using full precision with batch size {self.batch_size}")
                
                # 添加显存优化：启用梯度检查点以减少显存使用
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    print("Enabled gradient checkpointing to reduce VRAM usage")
                
                # 添加显存优化：设置CUDA内存分配策略
                torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
                torch.backends.cudnn.deterministic = False  # 允许非确定性算法以获得更好性能
                
                # 设置GPU内存分配策略
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()  # 清理GPU缓存
                
                print(f"GPU optimization: CUDA streams={len(self.streams)}, cuDNN benchmark enabled")
            
        def embed_images(self, image_paths):
            """嵌入图像列表"""
            if not image_paths:
                return np.zeros((0, 512), dtype="float32"), []
            
            # 简化预处理：直接使用多线程处理
            processed_tensors, valid_paths = preprocess_images_batch(
                image_paths, self.optimized_preprocess, max_workers=8
            )
            
            if not processed_tensors:
                return np.zeros((0, 512), dtype="float32"), []
            
            # 批量推理
            batch_size = min(self.batch_size, len(processed_tensors))
            all_embeddings = []
            
            for i in range(0, len(processed_tensors), batch_size):
                batch_tensors = processed_tensors[i:i+batch_size]
                
                # GPU优化：使用CUDA流和混合精度
                if self.is_gpu:
                    # 使用轮询的CUDA流进行并行处理
                    current_stream = self.streams[self.current_stream]
                    self.current_stream = (self.current_stream + 1) % len(self.streams)
                    
                    with torch.cuda.stream(current_stream):
                        # 使用混合精度推理
                        if self.mixed_precision:
                            with autocast():
                                with torch.no_grad():
                                    image_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
                                    image_features = self.model.encode_image(image_tensor)
                                    image_features = F.normalize(image_features, dim=-1)
                        else:
                            with torch.no_grad():
                                image_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
                                image_features = self.model.encode_image(image_tensor)
                                image_features = F.normalize(image_features, dim=-1)
                    
                    # 同步所有流以确保计算完成
                    for stream in self.streams:
                        torch.cuda.current_stream().wait_stream(stream)
                else:
                    # CPU模式
                    with torch.no_grad():
                        image_tensor = torch.stack(batch_tensors).to(self.device)
                        image_features = self.model.encode_image(image_tensor)
                        image_features = F.normalize(image_features, dim=-1)
                
                batch_embeddings = image_features.cpu().numpy().astype("float32")
                all_embeddings.append(batch_embeddings)
                
                # 释放内存
                del image_tensor, image_features
                if self.is_gpu and (i//batch_size+1) % 5 == 0:
                    torch.cuda.empty_cache()
            
            embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 512), dtype="float32")
            
            return embeddings, valid_paths
            
        def _preprocess_with_dataloader(self, image_paths):
            """使用DataLoader进行异步数据加载和预处理"""
            class ImageDataset(Dataset):
                def __init__(self, image_paths, transform):
                    self.image_paths = image_paths
                    self.transform = transform
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    try:
                        img_path = self.image_paths[idx]
                        image = Image.open(img_path).convert("RGB")
                        tensor = self.transform(image)
                        return tensor, img_path, True
                    except Exception as e:
                        # 返回一个空张量和错误标志
                        return torch.zeros(3, 224, 224), self.image_paths[idx], False
            
            # 创建数据集和数据加载器（GPU优化版本）
            dataset = ImageDataset(image_paths, self.optimized_preprocess)
            
            # GPU优化：根据设备类型调整数据加载器参数
            num_workers = min(8, self.num_processes) if self.is_gpu else min(4, self.num_processes)
            pin_memory = self.pin_memory and self.is_gpu
            
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                num_workers=num_workers,  # GPU模式下使用更多worker并行加载
                pin_memory=pin_memory,    # GPU模式下启用内存锁定
                persistent_workers=num_workers > 0,  # GPU模式下保持worker进程
                shuffle=False
            )
            
            processed_tensors = []
            valid_paths = []
            
            # 使用进度条显示加载进度
            print(f"Loading and preprocessing {len(image_paths)} images with DataLoader...")
            for batch_tensors, batch_paths, batch_valid in dataloader:
                # 只保留有效的图像
                for tensor, path, is_valid in zip(batch_tensors, batch_paths, batch_valid):
                    if is_valid:
                        processed_tensors.append(tensor)
                        valid_paths.append(path)
            
            return processed_tensors, valid_paths
            
        def _preprocess_with_multiprocessing(self, image_paths):
            """使用多进程进行图像预处理"""
            # 创建进程池
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                # 使用部分函数应用预处理函数
                process_func = partial(process_single_image, preprocess_func=self.optimized_preprocess)
                # 并行处理所有图像
                results = pool.map(process_func, image_paths)
            
            # 处理结果
            valid_paths = []
            processed_tensors = []
            
            for path, tensor, error in results:
                if error:
                    print(f"Warning: Could not process {path}: {error}")
                elif tensor is not None:
                    valid_paths.append(path)
                    processed_tensors.append(tensor)
            
            return processed_tensors, valid_paths
    
    return CLIPImageBackend(model, preprocess, optimized_preprocess, device, num_processes, 
                           batch_size=batch_size, mixed_precision=mixed_precision, pin_memory=pin_memory)

# 多进程处理单个图像的辅助函数
def process_single_image(path, preprocess_func):
    """处理单个图像的函数，用于多进程"""
    try:
        image = Image.open(path)
        tensor = preprocess_func(image)
        return path, tensor, None
    except Exception as e:
        return path, None, str(e)

@ImageEmbeddingBackend.register("clip_l14")
def create_clip_l14_image_backend(weights_path=None, device="auto", quantize=True, low_res=True, num_processes=4, 
                                 batch_size=None, mixed_precision=False, pin_memory=True):
    """CLIP ViT-L/14 backend for images with GPU acceleration and optimization options"""
    # lazy import to avoid hard dependency when using mock backend
    import clip
    # 增强的设备检测与自动切换
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # 打印设备信息
    if device == "cpu":
        print(f"Using CPU for inference")
    else:
        print(f"Using GPU for inference: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    if weights_path and os.path.exists(weights_path):
        model, preprocess = clip.load("ViT-L/14", device=device, download_root=weights_path)
    else:
        model, preprocess = clip.load("ViT-L/14", device=device)
    
    # 模型量化优化 - 仅在CPU模式下应用
    if device == "cpu" and quantize:
        print("Applying dynamic quantization to CLIP L14 model...")
        try:
            # 使用更安全的量化方法，避免嵌入层量化错误
            from torch.quantization import quantize_dynamic
            # 只量化线性层，避免嵌入层量化问题
            model = quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Model quantized successfully")
        except Exception as e:
            print(f"Quantization failed: {e}. Continuing without quantization.")
            # 如果量化失败，继续使用原始模型
    elif device.startswith("cuda"):
        # GPU优化：清理显存并预热模型
        torch.cuda.empty_cache()
        # 创建一个小批量数据预热模型
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input)
        print("GPU model warmed up successfully")
        
        # 添加显存优化：启用梯度检查点以减少显存使用
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing to reduce VRAM usage")
            
        # 添加显存优化：使用CUDA流水线并行化
        if torch.cuda.device_count() > 1:
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            print(f"Enabled DataParallel across {torch.cuda.device_count()} GPUs")
    
    # 创建优化的预处理函数
    if low_res:
        # 降低分辨率的预处理函数 - 使用224x224而不是112x112以匹配CLIP模型
        from torchvision import transforms
        optimized_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 保持224x224分辨率以匹配CLIP模型
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        optimized_preprocess = preprocess

    class CLIPImageBackend:
        def __init__(self, model, preprocess, optimized_preprocess, device, num_processes, 
                     batch_size=None, mixed_precision=False, pin_memory=True):
            self.model = model
            self.preprocess = preprocess
            self.optimized_preprocess = optimized_preprocess
            self.device = device
            self.num_processes = num_processes
            self.cache = {}  # 简单的缓存机制
            self.total_images = 0
            self.total_time = 0
            self.start_time = time.time()
            
            # GPU优化参数
            self.is_gpu = device.startswith("cuda")
            # GPU模式下使用更大的批处理大小
            self.batch_size = batch_size or (128 if self.is_gpu else 8)
            self.mixed_precision = mixed_precision and self.is_gpu  # 仅在GPU模式下启用混合精度
            self.pin_memory = pin_memory and self.is_gpu  # 仅在GPU模式下启用内存锁定
            
            # 如果使用GPU，创建CUDA流和混合精度scaler
            if self.is_gpu:
                # 创建多个CUDA流用于并行处理
                self.streams = [torch.cuda.Stream() for _ in range(2)]  # 两个流用于流水线处理
                self.current_stream = 0
                
                if self.mixed_precision:
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
                    print(f"Mixed precision enabled with batch size {self.batch_size}")
                else:
                    print(f"Using full precision with batch size {self.batch_size}")
                
                # 添加显存优化：启用梯度检查点以减少显存使用
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    print("Enabled gradient checkpointing to reduce VRAM usage")
                
                # 添加显存优化：设置CUDA内存分配策略
                torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
                torch.backends.cudnn.deterministic = False  # 允许非确定性算法以获得更好性能
                
                # 设置GPU内存分配策略
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()  # 清理GPU缓存
                
                print(f"GPU optimization: CUDA streams={len(self.streams)}, cuDNN benchmark enabled")
            
        def embed_images(self, image_paths):
            """嵌入图像列表"""
            if not image_paths:
                return np.zeros((0, 768), dtype="float32"), []
            
            # 简化预处理：直接使用多线程处理
            processed_tensors, valid_paths = preprocess_images_batch(
                image_paths, self.optimized_preprocess, max_workers=8
            )
            
            if not processed_tensors:
                return np.zeros((0, 768), dtype="float32"), []
            
            # 批量推理
            batch_size = min(self.batch_size, len(processed_tensors))
            all_embeddings = []
            
            for i in range(0, len(processed_tensors), batch_size):
                batch_tensors = processed_tensors[i:i+batch_size]
                
                # GPU优化：使用CUDA流和混合精度
                if self.is_gpu:
                    # 使用轮询的CUDA流进行并行处理
                    current_stream = self.streams[self.current_stream]
                    self.current_stream = (self.current_stream + 1) % len(self.streams)
                    
                    with torch.cuda.stream(current_stream):
                        # 使用混合精度推理
                        if self.mixed_precision:
                            with autocast():
                                with torch.no_grad():
                                    image_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
                                    image_features = self.model.encode_image(image_tensor)
                                    image_features = F.normalize(image_features, dim=-1)
                        else:
                            with torch.no_grad():
                                image_tensor = torch.stack(batch_tensors).to(self.device, non_blocking=True)
                                image_features = self.model.encode_image(image_tensor)
                                image_features = F.normalize(image_features, dim=-1)
                    
                    # 同步所有流以确保计算完成
                    for stream in self.streams:
                        torch.cuda.current_stream().wait_stream(stream)
                else:
                    # CPU模式
                    with torch.no_grad():
                        image_tensor = torch.stack(batch_tensors).to(self.device)
                        image_features = self.model.encode_image(image_tensor)
                        image_features = F.normalize(image_features, dim=-1)
                
                batch_embeddings = image_features.cpu().numpy().astype("float32")
                all_embeddings.append(batch_embeddings)
                
                # 释放内存
                del image_tensor, image_features
                if self.is_gpu and (i//batch_size+1) % 5 == 0:
                    torch.cuda.empty_cache()
            
            embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 768), dtype="float32")
            
            return embeddings, valid_paths
            
        def _preprocess_with_dataloader(self, image_paths):
            """使用DataLoader进行异步数据加载和预处理"""
            class ImageDataset(Dataset):
                def __init__(self, image_paths, transform):
                    self.image_paths = image_paths
                    self.transform = transform
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    try:
                        img_path = self.image_paths[idx]
                        image = Image.open(img_path).convert("RGB")
                        tensor = self.transform(image)
                        return tensor, img_path, True
                    except Exception as e:
                        # 返回一个空张量和错误标志
                        return torch.zeros(3, 224, 224), self.image_paths[idx], False
            
            # 创建数据集和数据加载器（GPU优化版本）
            dataset = ImageDataset(image_paths, self.optimized_preprocess)
            
            # GPU优化：根据设备类型调整数据加载器参数
            num_workers = min(8, self.num_processes) if self.is_gpu else min(4, self.num_processes)
            pin_memory = self.pin_memory and self.is_gpu
            
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                num_workers=num_workers,  # GPU模式下使用更多worker并行加载
                pin_memory=pin_memory,    # GPU模式下启用内存锁定
                persistent_workers=num_workers > 0,  # GPU模式下保持worker进程
                shuffle=False
            )
            
            processed_tensors = []
            valid_paths = []
            
            # 使用进度条显示加载进度
            print(f"Loading and preprocessing {len(image_paths)} images with DataLoader...")
            for batch_tensors, batch_paths, batch_valid in dataloader:
                # 只保留有效的图像
                for tensor, path, is_valid in zip(batch_tensors, batch_paths, batch_valid):
                    if is_valid:
                        processed_tensors.append(tensor)
                        valid_paths.append(path)
            
            return processed_tensors, valid_paths
            
        def _preprocess_with_multiprocessing(self, image_paths):
            """使用多进程进行图像预处理"""
            # 创建进程池
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                # 使用部分函数应用预处理函数
                process_func = partial(process_single_image, preprocess_func=self.optimized_preprocess)
                # 并行处理所有图像
                results = pool.map(process_func, image_paths)
            
            # 处理结果
            valid_paths = []
            processed_tensors = []
            
            for path, tensor, error in results:
                if error:
                    print(f"Warning: Could not process {path}: {error}")
                elif tensor is not None:
                    valid_paths.append(path)
                    processed_tensors.append(tensor)
            
            return processed_tensors, valid_paths
    
    return CLIPImageBackend(model, preprocess, optimized_preprocess, device, num_processes)

def preprocess_images_batch(image_paths, preprocess_func, max_workers=8):
    """多线程批量预处理图像"""
    def process_single(path):
        try:
            image = Image.open(path)
            tensor = preprocess_func(image)
            return path, tensor, None
        except Exception as e:
            return path, None, str(e)
    
    valid_paths = []
    processed_tensors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single, path): path for path in image_paths}
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                path, tensor, error = future.result()
                if error:
                    print(f"Warning: Could not process {path}: {error}")
                elif tensor is not None:
                    valid_paths.append(path)
                    processed_tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    return processed_tensors, valid_paths

def save_embeddings_mmap(embeddings, output_path):
    """使用内存映射保存大型嵌入向量"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取嵌入维度
    shape = embeddings.shape
    
    # 创建内存映射文件
    mmap_file = np.memmap(output_path, dtype='float32', mode='w+', shape=shape)
    
    # 分批写入数据以减少内存使用
    batch_size = min(1000, shape[0])
    for i in range(0, shape[0], batch_size):
        end_idx = min(i + batch_size, shape[0])
        mmap_file[i:end_idx] = embeddings[i:end_idx]
    
    # 确保数据写入磁盘
    mmap_file.flush()
    del mmap_file
    
    return output_path

def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="bounding_box_test")  # query or bounding_box_test
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--weights-path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--subset", choices=["gold", "full"], default="full")
    ap.add_argument("--filter-ids", type=str, default="", help="Filter gallery by person IDs from .npy/.txt/.csv")
    ap.add_argument("--workers", type=int, default=4, help="Number of worker threads for preprocessing")
    ap.add_argument("--quantize", action="store_true", help="Apply model quantization")
    ap.add_argument("--low-res", action="store_true", help="Use lower resolution images")
    ap.add_argument("--num-processes", type=int, default=None, help="Number of processes for multiprocessing")
    ap.add_argument("--use-mmap", action="store_true", help="Use memory mapping for large files")
    ap.add_argument("--paths-out", type=str, default="", help="Write processed image paths to a text file")
    ap.add_argument("--subset-count", type=int, default=10, help="Total images for gold subset (2 per ID)")
    args = ap.parse_args()
    
    # Load config if not provided via CLI
    import yaml
    with open("configs/reid.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 获取优化设置
    backend_name = args.backend or config.get("embed", {}).get("image_backend", "clip_b16")
    weights_path = args.weights_path or config.get("embed", {}).get("weights_path")
    
    # GPU加速优化：自动检测设备，优先使用GPU
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 根据设备类型优化批处理大小
    if device.startswith("cuda"):
        batch_size = min(args.batch_size, 128)  # GPU模式下使用更大的批处理大小
        print(f"GPU加速模式: 使用批处理大小 {batch_size}")
    else:
        batch_size = min(args.batch_size, 16)  # CPU模式下使用较小的批处理大小
        print(f"CPU模式: 使用批处理大小 {batch_size}")
    
    # 从配置文件获取优化选项
    optimization_config = config.get("embed", {}).get("optimization", {})
    quantize = args.quantize or optimization_config.get("quantize", True)
    low_res = args.low_res or optimization_config.get("low_res", True)
    num_processes = args.num_processes or optimization_config.get("num_processes", 4)
    use_mmap = args.use_mmap or optimization_config.get("use_mmap", True)
    
    print(f"Using optimized image backend: {backend_name}, device: {device}, batch_size: {batch_size}")
    print(f"Optimization settings: quantize={quantize}, low_res={low_res}, num_processes={num_processes}, use_mmap={use_mmap}")
    
    # 显示初始内存使用
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory usage: {mem_before:.1f} MB")
    
    # Initialize backend with optimization options
    backend = ImageEmbeddingBackend.get_backend(
        backend_name, 
        weights_path=weights_path, 
        device=device,
        quantize=quantize,
        low_res=low_res,
        num_processes=num_processes
    )
    
    # Find image files
    image_pattern = os.path.join(args.root, args.split, "*.jpg")
    image_paths = sorted(glob.glob(image_pattern))

    # Optional: filter gallery images to a provided ID list
    if args.filter_ids:
        def load_ids_generic(path):
            import csv
            ext = os.path.splitext(path)[1].lower()
            ids = []
            if ext == ".npy":
                arr = np.load(path, allow_pickle=True)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                ids = [str(x) for x in arr.tolist()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        reader = csv.reader(f)
                        rows = list(reader)
                    except Exception:
                        rows = [line.strip().split(",") for line in f.read().splitlines()]
                if rows:
                    header = rows[0]
                    if any(h.lower() == "id" for h in header):
                        idx = [i for i, h in enumerate(header) if h.lower() == "id"][0]
                        for r in rows[1:]:
                            if idx < len(r):
                                ids.append(str(r[idx]).strip())
                    else:
                        for r in rows:
                            if r:
                                ids.append(str(r[0]).strip())
            return set(ids)
        try:
            id_set = load_ids_generic(args.filter_ids)
            image_paths = [p for p in image_paths if os.path.basename(p).split("_")[0] in id_set]
            print(f"Filtered gallery to {len(image_paths)} images matching provided IDs")
        except Exception as e:
            print(f"[WARN] Failed to load filter IDs from {args.filter_ids}: {e}")

    if args.subset == "gold":
        # Configurable gold subset: select first K images with >=2 per ID
        from collections import defaultdict
        id_to_images = defaultdict(list)
        for path in image_paths:
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
        image_paths = gold_paths[:target_total]
    
    print(f"Found {len(image_paths)} images for processing")
    
    if not image_paths:
        print("Warning: No images found")
        # Create empty embeddings
        embeddings = np.zeros((0, 768), dtype="float32")
        all_valid_paths = []
    else:
        # 批处理并显示详细进度
        all_embeddings = []
        all_valid_paths = []
        total_batches = (len(image_paths) - 1) // batch_size + 1
        start_time = time.time()
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # 处理当前批次
            batch_embeds, valid_paths = backend.embed_images(batch_paths)
            
            if batch_embeds.size > 0:
                all_embeddings.append(batch_embeds)
                all_valid_paths.extend(valid_paths)
            
            # 详细进度跟踪
            current_batch = i // batch_size + 1
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / current_batch
            remaining_batches = total_batches - current_batch
            eta = avg_time_per_batch * remaining_batches
            
            # 计算当前处理速度
            current_speed = len(valid_paths) / avg_time_per_batch if avg_time_per_batch > 0 else 0
            
            # 显示内存使用情况
            mem_current = process.memory_info().rss / 1024 / 1024
            mem_diff = mem_current - mem_before
            
            print(f"Progress: {current_batch}/{total_batches} batches ({current_batch/total_batches*100:.1f}%)")
            print(f"ETA: {eta:.0f}s, Speed: {current_speed:.1f} img/s")
            print(f"Memory usage: {mem_current:.1f} MB (change: {mem_diff:+.1f} MB)")
            print(f"Processed {len(all_valid_paths)}/{len(image_paths)} images so far")
            
            # 定期释放内存
            if i % (batch_size * 5) == 0 and i > 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            embeddings = np.zeros((0, 768), dtype="float32")
        
        total_time = time.time() - start_time
        final_speed = len(all_valid_paths) / total_time if total_time > 0 else 0
        
        print(f"\n=== Performance Summary ===")
        print(f"Total processing time: {total_time:.2f}s for {len(all_valid_paths)} images")
        print(f"Average speed: {final_speed:.1f} img/s (target: 3-5 img/s)")
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        # 与基准速度比较
        baseline_speed = 1.4  # img/s
        speedup = final_speed / baseline_speed
        print(f"Speedup over baseline (1.4 img/s): {speedup:.1f}x")
    
    # 验证嵌入
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"
    assert not np.any(np.isinf(embeddings)), "Embeddings contain Inf values"
    
    print(f"Generated embeddings: {embeddings.shape}")
    
    # 保存嵌入
    if use_mmap and embeddings.shape[0] > 1000:
        # 对大型嵌入使用内存映射
        output_path = save_embeddings_mmap(embeddings, args.out)
        print(f"Saved large embeddings using memory mapping to {output_path}")
    else:
        # 对小型嵌入使用标准保存
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, embeddings)
        print(f"Saved embeddings to {args.out}")
    
    # 写出处理过的图像路径（如指定）
    try:
        if args.paths_out:
            os.makedirs(os.path.dirname(args.paths_out), exist_ok=True)
            with open(args.paths_out, 'w', encoding='utf-8') as f:
                for p in (all_valid_paths or []):
                    f.write(str(p) + '\n')
            print(f"Wrote processed image paths to {args.paths_out}")
    except Exception as e:
        print(f"Warning: failed to write paths_out: {e}")
    
    # 显示最终内存使用
    mem_final = process.memory_info().rss / 1024 / 1024
    mem_diff = mem_final - mem_before
    print(f"Final memory usage: {mem_final:.1f} MB (change from start: {mem_diff:+.1f} MB)")

if __name__ == "__main__":
    main()