import os, json, time
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset100.npy'
QUERY_LIST = ROOT/'larger_iso/64/query100.txt' if (ROOT/'larger_iso/64/query100.txt').exists() else ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt' if (ROOT/'larger_iso/64/gallery100.txt').exists() else ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
SEED=42
np.random.seed(SEED)

COLORS_11=['black','white','gray','red','orange','yellow','green','blue','purple','pink','brown']
COLOR_VARIANTS={
 'black':['black'], 'white':['white'], 'gray':['gray','grey','silver','charcoal'],
 'red':['red','dark red','bright red','scarlet','maroon','burgundy'],
 'orange':['orange','peach','apricot'], 'yellow':['yellow','gold','mustard'],
 'green':['green','dark green','light green','olive','lime','teal'],
 'blue':['blue','dark blue','light blue','navy','sky blue','azure'],
 'purple':['purple','violet','magenta'], 'pink':['pink','hot pink','light pink'],
 'brown':['brown','beige','khaki','tan','camel']
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

def encode_texts(model, texts):
    toks=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(toks)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

# windows

def center_square_crop(img):
    w,h=img.size; s=min(w,h); left=(w-s)//2; top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def top_bottom_crops(img):
    sq=center_square_crop(img)
    w,h=sq.size
    return sq.crop((0,0,w,h//2)), sq.crop((0,h//2,w,h))

def encode_gallery_window_embeddings(model, preprocess, g_paths):
    I_top=[]; I_full=[]; I_bottom=[]
    for p in g_paths:
        try:
            img=Image.open(p).convert('RGB')
            top_img, bot_img = top_bottom_crops(img)
            def enc(im):
                x=preprocess(im).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): v=model.encode_image(x)
                v=v/ v.norm(dim=-1, keepdim=True)
                return v.squeeze(0).cpu().numpy()
            I_top.append(enc(top_img))
            I_full.append(enc(center_square_crop(img)))
            I_bottom.append(enc(bot_img))
        except Exception:
            d=model.visual.output_dim
            I_top.append(np.zeros(d)); I_full.append(np.zeros(d)); I_bottom.append(np.zeros(d))
    return np.stack(I_top), np.stack(I_full), np.stack(I_bottom)

# explicit caps

def load_exp_caps(q_paths):
    if not EXPL_CAP_PATH.exists(): return {p:['a person'] for p in q_paths}
    try: data=json.loads(EXPL_CAP_PATH.read_text(encoding='utf-8'))
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

# query color labels

def latest_colorcheck_dir():
    dirs=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith('_attrbank_v4_colorcheck')]
    return sorted(dirs)[-1] if dirs else None

def load_query_color_labels():
    d=latest_colorcheck_dir()
    if d is None: return {}
    p=d/'query_color_labels.json'
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except: return {}

# attribute sentences from color labels

def build_attr_sents_from_labels(top_color, bottom_color, bottom_type='pants'):
    s=[]
    if top_color and top_color!='unknown':
        s.extend([f"a person wearing a {v} top" for v in COLOR_VARIANTS[top_color]])
    if bottom_color and bottom_color!='unknown' and bottom_type!='unknown':
        s.extend([f"a person wearing {v} {bottom_type}" for v in COLOR_VARIANTS[bottom_color]])
    return s[:16]

def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_triwin_weighted"; ensure_dir(out_dir)
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    model, preprocess = load_clip()
    # gallery window embeddings
    I_top, I_full, I_bottom = encode_gallery_window_embeddings(model, preprocess, g_paths)
    # explicit texts
    expl=load_exp_caps(q_paths)
    S_top_exp_rows=[]; S_full_exp_rows=[]; S_bot_exp_rows=[]
    for p in q_paths:
        sents=expl[p]
        V=encode_texts(model, sents)
        S_top_exp_rows.append(np.max(V@I_top.T, axis=0))
        S_full_exp_rows.append(np.max(V@I_full.T, axis=0))
        S_bot_exp_rows.append(np.max(V@I_bottom.T, axis=0))
    S_top_exp=np.vstack(S_top_exp_rows)
    S_full_exp=np.vstack(S_full_exp_rows)
    S_bot_exp=np.vstack(S_bot_exp_rows)
    Z_top_exp,_,_=global_zscore(S_top_exp); Z_full_exp,_,_=global_zscore(S_full_exp); Z_bot_exp,_,_=global_zscore(S_bot_exp)
    S_exp_tri = 0.6*Z_top_exp + 0.3*Z_full_exp + 0.1*Z_bot_exp
    np.save(out_dir/'S_exp_tri.npy', S_exp_tri)
    met_exp=rank1_map_ndcg(S_exp_tri, q_ids, g_ids)
    # attribute texts from color labels
    q_color_labels=load_query_color_labels()
    S_top_attr_mean_rows=[]; S_full_attr_mean_rows=[]; S_bot_attr_mean_rows=[]
    S_top_attr_max_rows=[]; S_full_attr_max_rows=[]; S_bot_attr_max_rows=[]
    for p in q_paths:
        lab=q_color_labels.get(basename_noext(p), {})
        qt=lab.get('top_color','unknown'); qb=lab.get('bottom_color','unknown')
        bottom_type='pants'
        s_top=[f"a person wearing a {v} top" for v in COLOR_VARIANTS.get(qt,[])] if qt!='unknown' else []
        s_bot=[f"a person wearing {v} {bottom_type}" for v in COLOR_VARIANTS.get(qb,[])] if qb!='unknown' else []
        # compute per window
        # top window
        if s_top:
            Vt=encode_texts(model, s_top); top_top=np.max(Vt@I_top.T, axis=0)
        else:
            top_top=np.zeros(len(g_paths))
        # full window
        full_top = np.max(encode_texts(model, s_top)@I_full.T, axis=0) if s_top else np.zeros(len(g_paths))
        full_bot = np.max(encode_texts(model, s_bot)@I_full.T, axis=0) if s_bot else np.zeros(len(g_paths))
        # bottom window
        if s_bot:
            Vb=encode_texts(model, s_bot); bot_bot=np.max(Vb@I_bottom.T, axis=0)
        else:
            bot_bot=np.zeros(len(g_paths))
        denom = (1 if s_top else 0)+(1 if s_bot else 0)
        mean_full = (full_top + full_bot)/ (denom if denom>0 else 1)
        max_full = np.maximum(full_top, full_bot)
        # collect
        S_top_attr_mean_rows.append(top_top)
        S_full_attr_mean_rows.append(mean_full)
        S_bot_attr_mean_rows.append(bot_bot)
        S_top_attr_max_rows.append(top_top)
        S_full_attr_max_rows.append(max_full)
        S_bot_attr_max_rows.append(bot_bot)
    S_top_attr_mean=np.vstack(S_top_attr_mean_rows)
    S_full_attr_mean=np.vstack(S_full_attr_mean_rows)
    S_bot_attr_mean=np.vstack(S_bot_attr_mean_rows)
    Z_top_attr_mean,_,_=global_zscore(S_top_attr_mean); Z_full_attr_mean,_,_=global_zscore(S_full_attr_mean); Z_bot_attr_mean,_,_=global_zscore(S_bot_attr_mean)
    S_attr_tri_mean = 0.6*Z_top_attr_mean + 0.3*Z_full_attr_mean + 0.1*Z_bot_attr_mean
    np.save(out_dir/'S_attr_tri_mean.npy', S_attr_tri_mean)
    met_attr_mean=rank1_map_ndcg(S_attr_tri_mean, q_ids, g_ids)
    # max version
    S_top_attr_max=np.vstack(S_top_attr_max_rows)
    S_full_attr_max=np.vstack(S_full_attr_max_rows)
    S_bot_attr_max=np.vstack(S_bot_attr_max_rows)
    Z_top_attr_max,_,_=global_zscore(S_top_attr_max); Z_full_attr_max,_,_=global_zscore(S_full_attr_max); Z_bot_attr_max,_,_=global_zscore(S_bot_attr_max)
    S_attr_tri_max = 0.6*Z_top_attr_max + 0.3*Z_full_attr_max + 0.1*Z_bot_attr_max
    np.save(out_dir/'S_attr_tri_max.npy', S_attr_tri_max)
    met_attr_max=rank1_map_ndcg(S_attr_tri_max, q_ids, g_ids)
    # summary
    write_json(out_dir/'metrics.json', {'exp_triwin':met_exp,'attr_triwin_mean':met_attr_mean,'attr_triwin_max':met_attr_max})
    write_csv(out_dir/'summary.csv', [
        ['attrbank_v4_triwin_exp', met_exp['Rank-1'], met_exp['mAP'], met_exp['nDCG@10'], 'fixed-weight','none'],
        ['attrbank_v4_triwin_attr_mean', met_attr_mean['Rank-1'], met_attr_mean['mAP'], met_attr_mean['nDCG@10'], 'fixed-weight','none'],
        ['attrbank_v4_triwin_attr_max', met_attr_max['Rank-1'], met_attr_max['mAP'], met_attr_max['nDCG@10'], 'fixed-weight','none'],
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    write_json(out_dir/'config.yaml', {'weights':[0.6,0.3,0.1],'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},'seed':SEED})
    Path(out_dir/'README.md').write_text('v4三窗加权：显式路max-over-sents对三窗图像嵌入做加权，属性路基于colorcheck的top/bottom句分别对三窗，输出S_exp_tri与S_attr_tri_{mean,max}。', encoding='utf-8')

if __name__=='__main__':
    main()