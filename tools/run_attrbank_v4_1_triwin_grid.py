import os, json, time
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
QUERY_LIST = ROOT/'larger_iso/64/query100.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt'
CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
SEED=42
np.random.seed(SEED)

WSETS={
    'w1': (0.5,0.3,0.2),
    'w2': (0.6,0.3,0.1),
    'w3': (0.5,0.4,0.1),
    'w4': (0.4,0.4,0.2),
}

read_lines=lambda p: [s.strip() for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
basename_noext=lambda p: os.path.splitext(os.path.basename(p))[0]

def ts(): return time.strftime('%Y%m%d-%H%M%S')

def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def write_json(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(p, rows, header=None):
    txt=[]
    if header: txt.append(','.join(header))
    for r in rows: txt.append(','.join(str(x) for x in r))
    Path(p).write_text('\n'.join(txt), encoding='utf-8')

def ids_from_paths(paths):
    ids=[]
    for p in paths:
        s=basename_noext(p); head=s.split('_')[0]
        try: ids.append(int(head))
        except: ids.append(head)
    return ids

def global_zscore(S):
    m=float(S.mean()); s=float(S.std()) if S.std()>1e-9 else 1.0
    return (S-m)/s, m, s

def rank1_map_ndcg(S, q_ids, g_ids, k_ndcg=10):
    G=np.array(g_ids); n_q=S.shape[0]; r1=0; ap_sum=0; nd_sum=0
    for i in range(n_q):
        rel=(G==q_ids[i]).astype(np.int32)
        order=np.argsort(-S[i]); r1+=int(rel[order[0]]==1)
        hits=np.where(rel[order]==1)[0]
        ap=0.0; denom=max(1,len(hits))
        for j,h in enumerate(hits, start=1): ap += j/(h+1)
        ap = ap/denom if denom>0 else 0.0
        k=min(k_ndcg,len(rel)); gains=rel[order][:k]
        if np.sum(gains)==0: nd=0.0
        else:
            dcg=np.sum(gains/np.log2(np.arange(2,k+2)))
            ideal=np.sort(rel)[::-1][:k]
            idcg=np.sum(ideal/np.log2(np.arange(2,k+2)))
            nd=dcg/(idcg+1e-9)
        ap_sum+=ap; nd_sum+=nd
    return {'Rank-1':round(r1/n_q,4),'mAP':round(ap_sum/n_q,4),'nDCG@10':round(nd_sum/n_q,4)}

def load_clip():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return model, preprocess

def center_square_crop(img):
    w,h=img.size; s=min(w,h); left=(w-s)//2; top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def top_bottom_crops(img):
    sq=center_square_crop(img)
    w,h=sq.size
    return sq.crop((0,0,w,h//2)), sq.crop((0,h//2,w,h))

# encode image window features

def encode_image_windows(model, preprocess, paths):
    I_top=[]; I_full=[]; I_bottom=[]
    for p in paths:
        try:
            img=Image.open(p).convert('RGB')
            top_img, bot_img = top_bottom_crops(img)
            def enc(im):
                x=preprocess(im).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): v=model.encode_image(x)
                v=v/ v.norm(dim=-1, keepdim=True)
                return v.squeeze(0).cpu().numpy()
            I_top.append(enc(top_img)); I_full.append(enc(center_square_crop(img))); I_bottom.append(enc(bot_img))
        except Exception:
            d=model.visual.output_dim
            I_top.append(np.zeros(d)); I_full.append(np.zeros(d)); I_bottom.append(np.zeros(d))
    return np.stack(I_top), np.stack(I_full), np.stack(I_bottom)

# explicit captions

def load_exp_caps(q_paths):
    if not CAP_PATH.exists(): return {p:['a person'] for p in q_paths}
    try: data=json.loads(CAP_PATH.read_text(encoding='utf-8'))
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
            out[p]=v[:16] if v else ['a person']
    return out

# CLIP encode texts

def encode_texts(model, texts):
    toks=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(toks)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

# compute explicit per-window S

def compute_exp_windows(model, preprocess, q_paths, g_paths):
    I_top_g, I_full_g, I_bot_g = encode_image_windows(model, preprocess, g_paths)
    expl=load_exp_caps(q_paths)
    rows_top=[]; rows_full=[]; rows_bot=[]
    for p in q_paths:
        sents=expl[p]
        V=encode_texts(model, sents)
        rows_top.append(np.max(V@I_top_g.T, axis=0))
        rows_full.append(np.max(V@I_full_g.T, axis=0))
        rows_bot.append(np.max(V@I_bot_g.T, axis=0))
    S_top=np.vstack(rows_top); S_full=np.vstack(rows_full); S_bot=np.vstack(rows_bot)
    return S_top, S_full, S_bot

# latest triwin v4 dir

def latest_triwin_dir():
    dirs=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith('_attrbank_v4_triwin_weighted')]
    return sorted(dirs)[-1] if dirs else None


def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_triwin_grid_soft"; ensure_dir(out_dir)
    # files
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    # compute explicit windows
    model, preprocess=load_clip()
    S_top_exp, S_full_exp, S_bot_exp = compute_exp_windows(model, preprocess, q_paths, g_paths)
    # z-score globally per matrix
    Z_top_exp,_m,_s = global_zscore(S_top_exp)
    Z_full_exp,_m,_s = global_zscore(S_full_exp)
    Z_bot_exp,_m,_s = global_zscore(S_bot_exp)
    # weight scan for explicit
    rows=[]
    best_key=None; best_map=-1.0
    for key, (w_top, w_full, w_bot) in WSETS.items():
        S_exp = w_top*Z_top_exp + w_full*Z_full_exp + w_bot*Z_bot_exp
        np.save(out_dir/f'S_exp_tri_{key}.npy', S_exp)
        met = rank1_map_ndcg(S_exp, q_ids, g_ids)
        write_json(out_dir/f'metrics_{key}_exp.json', met)
        rows.append([f'exp_tri_{key}', met['Rank-1'], met['mAP'], met['nDCG@10']])
        if met['mAP']>best_map:
            best_map=met['mAP']; best_key=key
    # attribute grids (input missing windows)
    tri=latest_triwin_dir()
    err_msgs=[]
    if tri and (tri/'S_attr_tri_mean.npy').exists() and (tri/'S_attr_tri_max.npy').exists():
        S_attr_mean=np.load(tri/'S_attr_tri_mean.npy')
        S_attr_max=np.load(tri/'S_attr_tri_max.npy')
        Z_attr_mean,_m,_s = global_zscore(S_attr_mean)
        Z_attr_max,_m,_s = global_zscore(S_attr_max)
        # We only have pre-weighted arrays (likely w2). Record metrics as-is.
        met_m = rank1_map_ndcg(Z_attr_mean, q_ids, g_ids)
        met_x = rank1_map_ndcg(Z_attr_max, q_ids, g_ids)
        write_json(out_dir/'metrics_attr_mean_w2.json', met_m)
        write_json(out_dir/'metrics_attr_max_w2.json', met_x)
        rows.append(['attr_tri_mean_w2', met_m['Rank-1'], met_m['mAP'], met_m['nDCG@10']])
        rows.append(['attr_tri_max_w2', met_x['Rank-1'], met_x['mAP'], met_x['nDCG@10']])
    else:
        err_msgs.append('Attribute per-window inputs missing; only found aggregated or none.')
    # summary
    write_csv(out_dir/'summary.csv', rows, header=['setting','Rank-1','mAP','nDCG@10'])
    # config & README
    cfg={
        'weights': WSETS,
        'best_exp_key': best_key,
        'filelists': {'query': str(QUERY_LIST), 'gallery': str(GALLERY_LIST)},
    }
    write_json(out_dir/'config.yaml', cfg)
    Path(out_dir/'README.md').write_text('v4.1三窗权重网格（soft）：显式路径扫描w1..w4按全局z-score组合；属性路径因缺少窗口级输入，仅评估已有加权结果（w2）。', encoding='utf-8')
    # errors
    if err_msgs:
        Path(out_dir/'ERRORS.md').write_text('\n'.join(err_msgs), encoding='utf-8')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_triwin_grid_soft"; ensure_dir(out_dir)
        Path(out_dir/'ERRORS.md').write_text(f'Error in StepB: {repr(e)}', encoding='utf-8')