#!/usr/bin/env python
import json, time
from pathlib import Path
import numpy as np
import clip, torch
from PIL import Image

ROOT = Path('d:/PRP SunnyLab/ReID')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
QUERY_LIST = ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'

CAUSES = ['颜色混淆','包缺失','包误检','鞋颜色混淆']

COLORS_11 = ['black','white','blue','red','green','yellow','purple','pink','orange','brown','gray']

def ts(): return time.strftime('%Y%m%d-%H%M%S', time.localtime())

def read_lines(p): return [l.strip() for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

def basename_noext(p): return Path(p).stem

def load_exp_caps(q_paths):
    if not EXPL_CAP_PATH.exists(): return {p:['a person'] for p in q_paths}
    try: data=json.loads(Path(EXPL_CAP_PATH).read_text(encoding='utf-8'))
    except: return {p:['a person'] for p in q_paths}
    by_base={}
    if isinstance(data,dict):
        for k,v in data.items(): by_base[basename_noext(k)]=v
    elif isinstance(data,list):
        for it in data:
            if isinstance(it,dict):
                fn=it.get('file') or it.get('path') or it.get('image') or ''
                txts=it.get('captions') or it.get('text') or it.get('desc') or it.get('salient') or []
                by_base[basename_noext(fn)]=txts
    out={}
    for p in q_paths:
        v=by_base.get(basename_noext(p));
        if not v: out[p]=['a person']
        else:
            if isinstance(v,str): v=[v]
            v=[s for s in v if isinstance(s,str) and s.strip()]
            out[p]=v if v else ['a person']
    return out

def parse_basic_attrs(caption):
    t=caption.lower()
    tokens=t.replace(',',' ').replace('.',' ').split()
    # colors
    top_col='unknown'
    for w in COLORS_11:
        if w in tokens: top_col=w; break
    # bag
    bag_present = any(w in tokens for w in ['backpack','bag','handbag','shoulder','tote'])
    bag_absent = ('no' in tokens and 'bag' in tokens) or ('without' in tokens and 'bag' in tokens)
    # shoes
    shoe_word = any(w in tokens for w in ['shoe','shoes','sneaker','sneakers','boots'])
    shoes_color = 'white' if ('white' in tokens and shoe_word) else ('black' if ('black' in tokens and shoe_word) else ('other' if shoe_word else 'unknown'))
    return dict(top_color=top_col, bag_present=bag_present, bag_absent=bag_absent, shoes_color=shoes_color)

def load_clip():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return model, preprocess

def l2norm(x):
    d = np.linalg.norm(x, axis=1, keepdims=True); d[d==0]=1.0
    return x/d

def encode_texts(model, texts):
    toks = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        v = model.encode_text(toks).float().cpu().numpy()
    return l2norm(v)

def ids_from_paths(paths):
    ids=[]
    for p in paths:
        s=basename_noext(p)
        head=s.split('_')[0]
        try: ids.append(int(head))
        except: ids.append(head)
    return ids

def predict_gallery_attrs(model, I_base):
    # build prompts
    color_prompts = [f"a person wearing a {c} top" for c in COLORS_11]
    bag_prompts = ["a person with a backpack", "no bag"]
    shoes_prompts = ["a person wearing white shoes", "a person wearing black shoes", "a person wearing colorful shoes"]
    T_color = encode_texts(model, color_prompts)
    T_bag = encode_texts(model, bag_prompts)
    T_shoes = encode_texts(model, shoes_prompts)
    S_color = T_color @ I_base.T
    S_bag = T_bag @ I_base.T
    S_shoes = T_shoes @ I_base.T
    pred = []
    for j in range(I_base.shape[0]):
        c_idx = int(np.argmax(S_color[:, j])); c = COLORS_11[c_idx]
        b_idx = int(np.argmax(S_bag[:, j])); b = 'with' if b_idx==0 else 'no'
        s_idx = int(np.argmax(S_shoes[:, j])); s = ['white','black','other'][s_idx]
        pred.append(dict(top_color=c, bag=b, shoes_color=s))
    return pred

def choose_latest_lr_dir():
    cmp = ROOT/'outputs/comparison'
    cand = sorted([p for p in cmp.glob('*_attrbank_v3_lr') if p.is_dir()])
    if not cand: raise FileNotFoundError('No attrbank_v3_lr outputs found')
    return cand[-1]

def main():
    out_dir = ROOT/'outputs/comparison'/f"{ts()}_attrbank_v3_error"
    out_dir.mkdir(parents=True, exist_ok=True)
    # paths and data
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    expl_caps = load_exp_caps(q_paths)
    model, preprocess = load_clip()
    I_base = np.load(EMB_IMAGE_PATH) if EMB_IMAGE_PATH.exists() else None
    if I_base is None:
        # encode gallery if needed
        ims=[]
        for p in g_paths:
            im = Image.open(p).convert('RGB')
            ims.append(preprocess(im))
        with torch.no_grad():
            I_base = model.encode_image(torch.stack(ims).to(DEVICE)).float().cpu().numpy()
        I_base = l2norm(I_base)
    # predict gallery attrs
    g_attr = predict_gallery_attrs(model, I_base)
    # load fused scores
    lr_dir = choose_latest_lr_dir()
    S = np.load(lr_dir/'S_fuse_lr.npy')
    rows = []
    cause_counts = {c:0 for c in CAUSES}
    for i, q in enumerate(q_paths):
        order = np.argsort(-S[i])
        qid = q_ids[i]
        q_attr = parse_basic_attrs(expl_caps[q][0])
        taken = 0
        for rank in order:
            if g_ids[rank] == qid:
                continue
            # failure case
            ga = g_attr[rank]
            causes = []
            if q_attr['top_color']!='unknown' and ga['top_color']!=q_attr['top_color']:
                causes.append('颜色混淆')
            if q_attr['bag_present'] and ga['bag']=='no':
                causes.append('包缺失')
            if q_attr['bag_absent'] and ga['bag']=='with':
                causes.append('包误检')
            if q_attr['shoes_color']!='unknown' and ga['shoes_color']!=q_attr['shoes_color']:
                causes.append('鞋颜色混淆')
            cause = '|'.join(causes) if causes else '未知'
            for c in causes: cause_counts[c]+=1
            rows.append([q, qid, taken+1, g_paths[rank], g_ids[rank], float(S[i, rank]), cause])
            taken += 1
            if taken >= 20:
                break
    # write outputs
    import csv
    with open(out_dir/'top20_failures.csv', 'w', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['query_path','query_id','fail_rank','gallery_path','gallery_id','score','cause'])
        for r in rows: w.writerow(r)
    # README summary
    summary = '\n'.join([f"{k}: {v}" for k,v in cause_counts.items()])
    (out_dir/'README.md').write_text('Top-20 失败样本原因统计:\n' + summary, encoding='utf-8')
    print(f"Wrote {out_dir/'top20_failures.csv'} and README.md with cause summary")

if __name__=='__main__':
    main()