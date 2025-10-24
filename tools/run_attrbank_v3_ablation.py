import os, json, time, pickle
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    SK=True
except Exception:
    SK=False

ROOT = Path('d:/PRP SunnyLab/ReID')
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
QUERY_LIST = ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
np.random.seed(42)

COLORS_11 = ['black','white','blue','red','green','yellow','purple','pink','orange','brown','gray']
COLOR_MAP = {'navy':'blue','dark blue':'blue','light blue':'blue','azure':'blue','sky':'blue','grey':'gray','silver':'gray','charcoal':'gray','dark grey':'gray','blonde':'yellow','gold':'yellow','maroon':'red','burgundy':'red','wine':'red','beige':'brown','khaki':'brown','tan':'brown','camel':'brown','violet':'purple','magenta':'purple','lime':'green','olive':'green','teal':'green','peach':'orange','apricot':'orange'}

def ts(): return time.strftime('%Y%m%d-%H%M%S', time.localtime())

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def read_lines(p): return [l.strip() for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

def write_json(p, o): Path(p).write_text(json.dumps(o, indent=2), encoding='utf-8')

def write_csv(p, rows, header=None):
    import csv
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); 
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def basename_noext(p): return Path(p).stem

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

def encode_images(model, preprocess, paths):
    ims=[]
    for p in paths:
        try: im=Image.open(p).convert('RGB')
        except: im=Image.new('RGB',(224,224),(127,127,127))
        ims.append(preprocess(im))
    batch=torch.stack(ims).to(DEVICE)
    with torch.no_grad(): v=model.encode_image(batch).float().cpu().numpy()
    return l2norm(v)

def global_zscore(M):
    m = M.mean(); s=M.std()+1e-9
    return (M-m)/s

def rank1_map_ndcg(S, q_ids, g_ids, k_ndcg=10):
    n_q=S.shape[0]; r1=0; ap_sum=0.; nd_sum=0.
    G=np.array(g_ids)
    for i in range(n_q):
        order=np.argsort(-S[i]); rel=(G[order]==q_ids[i]).astype(np.int32)
        r1 += int(rel[0]==1)
        pos=np.where(rel==1)[0]
        ap = 0. if len(pos)==0 else float(np.mean([np.sum(rel[:idx+1])/(idx+1) for idx in pos]))
        ap_sum+=ap
        k=min(k_ndcg,len(rel)); gains=rel[:k]
        if np.sum(gains)==0: nd=0.
        else:
            dcg=np.sum(gains/np.log2(np.arange(2,k+2)))
            ideal=np.sort(rel)[::-1][:k]
            idcg=np.sum(ideal/np.log2(np.arange(2,k+2)))
            nd=dcg/(idcg+1e-9)
        nd_sum+=nd
    return {'Rank-1':round(r1/n_q,4),'mAP':round(ap_sum/n_q,4),'nDCG@10':round(nd_sum/n_q,4)}

def ids_from_paths(paths):
    ids=[]
    for p in paths:
        s=basename_noext(p)
        head=s.split('_')[0]
        try: ids.append(int(head))
        except: ids.append(head)
    return ids

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
            out[p]=v if v else ['a person']
    return out

def normalize_color(tok):
    t=tok.lower().strip()
    if t in COLORS_11: return t
    return COLOR_MAP.get(t, t if t in COLORS_11 else 'unknown')

def parse_basic_attrs(caption):
    t=caption.lower()
    tokens=t.replace(',',' ').replace('.',' ').split()
    # colors
    top_col='unknown'; bottom_col='unknown'
    for w in COLORS_11 + list(COLOR_MAP.keys()):
        if w in tokens:
            c=normalize_color(w); 
            if top_col=='unknown': top_col=c
            elif bottom_col=='unknown': bottom_col=c
    # bottom type
    if 'skirt' in tokens: bottom_type='skirt'
    elif any(w in tokens for w in ['pant','pants','trouser','trousers','jeans','shorts','leggings']): bottom_type='pants'
    else: bottom_type='unknown'
    # bag
    bag_present = any(w in tokens for w in ['backpack','bag','handbag','shoulder','tote'])
    bag_absent = ('no' in tokens and 'bag' in tokens) or ('without' in tokens and 'bag' in tokens)
    # shoes
    shoe_word = any(w in tokens for w in ['shoe','shoes','sneaker','sneakers','boots'])
    shoes_color = 'white' if ('white' in tokens and shoe_word) else ('black' if ('black' in tokens and shoe_word) else ('other' if shoe_word else 'unknown'))
    return dict(top_color=top_col, bottom_color=bottom_col, bottom_type=bottom_type, bag_present=bag_present, bag_absent=bag_absent, shoes_color=shoes_color)

def build_attr_sentences_v3_flags(cap_list, include):
    base=cap_list[0] if cap_list else 'a person'
    a=parse_basic_attrs(base)
    s=[]
    if include.get('color',True):
        if include.get('bottom',True):
            if a['top_color']!='unknown': s.append(f"a person wearing a {a['top_color']} top")
            if a['bottom_type']!='unknown':
                bc=a['bottom_color'] if a['bottom_color']!='unknown' else ''
                if a['bottom_type']=='skirt': s.append(f"a person wearing {bc+' ' if bc else ''}skirt")
                else: s.append(f"a person wearing {bc+' ' if bc else ''}pants")
            elif a['bottom_color']!='unknown':
                s.append(f"a person wearing {a['bottom_color']} pants")
        else:
            # bottom disabled: only top color
            if a['top_color']!='unknown': s.append(f"a person wearing a {a['top_color']} top")
    else:
        # color off: only type if any and bottom allowed
        if include.get('bottom',True) and a['bottom_type']!='unknown':
            if a['bottom_type']=='skirt': s.append(f"a person wearing skirt")
            else: s.append(f"a person wearing pants")
    if include.get('bag',True):
        if a['bag_present']: s.append("a person with a backpack")
        elif a['bag_absent']: s.append("no bag")
    if include.get('shoes',True):
        if a['shoes_color'] in ['white','black','other']: s.append(f"a person wearing {a['shoes_color']} shoes")
    return s[:8]

def fit_lr_5fold(S_exp, S_attr):
    Z1 = global_zscore(S_exp)
    Z2 = global_zscore(S_attr)
    X = np.stack([Z1.flatten(), Z2.flatten()], axis=1)
    return kfold_predict(X, S_exp.shape)

def kfold_predict(X, shape):
    # build labels
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    G=np.array(g_ids)
    y=[]
    for qid in q_ids: y.extend((G==qid).astype(np.int32).tolist())
    y=np.array(y)
    yhat=np.zeros_like(y,dtype=float)
    if not SK:
        w=np.zeros(X.shape[1]); b=0.; lr_rate=0.05
        for _ in range(600):
            z=X@w+b; p=1/(1+np.exp(-z)); w-=lr_rate*(X.T@(p-y)/len(y)); b-=lr_rate*(np.sum(p-y)/len(y))
        yhat = 1/(1+np.exp(-(X@w+b)))
        return yhat.reshape(shape)
    # sklearn path
    pos_cnt = int(np.sum(y))
    if pos_cnt < 2:
        # degenerate: train global LR and return proba
        lr=LogisticRegression(max_iter=2000,class_weight='balanced'); lr.fit(X, y)
        p = lr.predict_proba(X)[:,1]
        return p.reshape(shape)
    n_splits = 5 if pos_cnt>=5 else max(2, pos_cnt)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lr_full=None
    for tr, va in skf.split(X, y):
        if len(np.unique(y[tr]))<2:
            if lr_full is None:
                lr_full=LogisticRegression(max_iter=2000,class_weight='balanced'); lr_full.fit(X, y)
            p = lr_full.predict_proba(X[va])[:,1]
            yhat[va]=p
            continue
        lr=LogisticRegression(max_iter=2000,class_weight='balanced'); lr.fit(X[tr], y[tr])
        p = lr.predict_proba(X[va])[:,1]
        if len(np.unique(y[va]))>=2:
            pl=LogisticRegression(max_iter=500)
            pl.fit(p.reshape(-1,1), y[va])
            yhat[va]=pl.predict_proba(p.reshape(-1,1))[:,1]
        else:
            yhat[va]=p
    return yhat.reshape(shape)

def main():
    ts_tag = ts()
    out_dir = ROOT/'outputs/comparison'/f"{ts_tag}_attrbank_v3_ablation"
    ensure_dir(out_dir)

    # data & model
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    model, preprocess = load_clip()
    if EMB_IMAGE_PATH.exists():
        I_base = np.load(EMB_IMAGE_PATH)
    else:
        ensure_dir(EMB_IMAGE_PATH.parent)
        I_base = encode_images(model, preprocess, g_paths); np.save(EMB_IMAGE_PATH, I_base)

    # explicit
    expl_caps = load_exp_caps(q_paths)
    expl_texts = [expl_caps[p][0] for p in q_paths]
    T_exp = encode_texts(model, expl_texts)
    S_exp = T_exp @ I_base.T

    # variants
    variants = [
        ('mean_base', {'color':True,'bottom':True,'bag':True,'shoes':True}, 'mean'),
        ('max_base', {'color':True,'bottom':True,'bag':True,'shoes':True}, 'max'),
        ('no_color_mean', {'color':False,'bottom':True,'bag':True,'shoes':True}, 'mean'),
        ('no_bottom_mean', {'color':True,'bottom':False,'bag':True,'shoes':True}, 'mean'),
        ('no_bag_mean', {'color':True,'bottom':True,'bag':False,'shoes':True}, 'mean'),
        ('no_shoes_mean', {'color':True,'bottom':True,'bag':True,'shoes':False}, 'mean'),
        ('no_color_max', {'color':False,'bottom':True,'bag':True,'shoes':True}, 'max'),
        ('no_bottom_max', {'color':True,'bottom':False,'bag':True,'shoes':True}, 'max'),
        ('no_bag_max', {'color':True,'bottom':True,'bag':False,'shoes':True}, 'max'),
        ('no_shoes_max', {'color':True,'bottom':True,'bag':True,'shoes':False}, 'max'),
    ]
    rows=[]
    for name, flags, agg in variants:
        attr_texts = [build_attr_sentences_v3_flags(expl_caps[p], flags) for p in q_paths]
        S_rows=[]
        for sents in attr_texts:
            if len(sents)==0:
                S_rows.append(np.zeros(len(g_paths)))
                continue
            V=encode_texts(model, sents); S_tmp=V@I_base.T
            S_rows.append(np.mean(S_tmp,axis=0) if agg=='mean' else np.max(S_tmp,axis=0))
        S_attr = np.vstack(S_rows)
        S_lr = fit_lr_5fold(S_exp, S_attr)
        met = rank1_map_ndcg(S_lr, q_ids, g_ids)
        rows.append([name, met['Rank-1'], met['mAP'], met['nDCG@10']])
        write_json(out_dir/f"metrics_{name}.json", met)
    write_csv(out_dir/'summary.csv', rows, header=['variant','Rank-1','mAP','nDCG@10'])
    Path(out_dir/'README.md').write_text('v3 消融：mean/max 与类别去除（颜色/下装/包/鞋）。指标为 LR 融合后的表现。', encoding='utf-8')

if __name__=='__main__':
    main()