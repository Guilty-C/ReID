#!/usr/bin/env python
import os
import json
import subprocess
import time
import numpy as np

def main():
    workdir = os.getcwd()
    dataset_root = os.getenv('DATASET_ROOT', os.path.join(workdir, 'data', 'mock_market'))
    out_dir = os.path.join(workdir, 'outputs', 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    combos = [('clip_l14', 'clip_l14'), ('clip_l14', 'clip_b16'), 
              ('clip_b16', 'clip_l14'), ('clip_b16', 'clip_b16')]
    
    with open(os.path.join(out_dir, 'metrics.csv'), 'w') as f:
        f.write('text_backend,image_backend,rank1,mAP,n_query,n_gallery,elapsed_s\n')

    for tb, ib in combos:
        run_dir = os.path.join(out_dir, f'{tb}_{ib}')
        os.makedirs(run_dir, exist_ok=True)
        start_time = time.time()
        
        try:
            # 运行端到端流程
            subprocess.run(['python', 'tools/captioner.py', '--root', dataset_root, 
                          '--out', os.path.join(run_dir, 'captions.json'), '--mode', 'json'], check=True)
            subprocess.run(['python', 'tools/embed_text.py', '--captions', 
                          os.path.join(run_dir, 'captions.json'), '--out', os.path.join(run_dir, 'text_embeds.npy')], check=True)
            subprocess.run(['python', 'tools/embed_image.py', '--root', dataset_root, 
                          '--out', os.path.join(run_dir, 'img_embeds.npy')], check=True)
            subprocess.run(['python', 'tools/retrieve_eval.py', '--text', 
                          os.path.join(run_dir, 'text_embeds.npy'), '--img', 
                          os.path.join(run_dir, 'img_embeds.npy'), '--out', 
                          os.path.join(run_dir, 'metrics.json')], check=True)
            
            elapsed = time.time() - start_time
            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            
            with open(os.path.join(out_dir, 'metrics.csv'), 'a') as f:
                f.write(f'{tb},{ib},{metrics["rank1"]},{metrics["mAP"]},{metrics["n_query"]},{metrics["n_gallery"]},{elapsed:.2f}\n')
            
            print(f'[ABLATE] 完成组合: {tb} + {ib}, 耗时: {elapsed:.2f}s')
            
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] 组合 {tb} + {ib} 运行失败: {e}')
            continue

    print('[ABLATE] 消融实验完成')

if __name__ == '__main__':
    main()