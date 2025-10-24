import os, json, time, pickle, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import clip

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    SKLEARN_OK=True
except Exception:
    SKLEARN_OK=False

ROOT = Path('d:/PRP SunnyLab/ReID')
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
QUERY_LIST = ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
LABELS_PATH = ROOT/'data/market1501.images.labels_subset50.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
np.random.seed(42)

COLORS_11 = ['black','white','blue','red','green','yellow','purple','pink','orange','brown','gray']
COLOR_MAP = {
    'navy':'blue','dark blue':'blue','light blue':'blue','azure':'blue','sky':'blue',
    'grey':'gray','silver':'gray','charcoal':'gray','dark grey':'gray',
    'blonde':'yellow','gold':'yellow',
    'maroon':'red','burgundy':'red','wine':'red',
    'beige':'brown','khaki':'brown','tan':'brown','camel':'brown',
    'violet':'purple','magenta':'purple',
    'lime':'green','olive':'green','teal':'green',
    'peach':'orange','apricot':'orange'
}

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
    return (M-m)/s, float(m), float(s)

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

def get_qg():
    q=read_lines(QUERY_LIST); g=read_lines(GALLERY_LIST); return q,g

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
    # top color
    top_col='unknown'
    for w in COLORS_11 + list(COLOR_MAP.keys()):
        if w in tokens:
            c=normalize_color(w); top_col=c; break
    # bottom
    if 'skirt' in tokens: bot_type='skirt'
    elif any(w in tokens for w in ['pant','pants','trouser','trousers','jeans','shorts','leggings']): bot_type='pants'
    else: bot_type='unknown'
    bot_col='unknown'
    for w in COLORS_11 + list(COLOR_MAP.keys()):
        if w in tokens:
            c=normalize_color(w); bot_col=c; break
    # bag
    bag_present = any(w in tokens for w in ['backpack','bag','handbag','shoulder','tote'])
    bag_absent = ('no' in tokens and 'bag' in tokens) or ('without' in tokens and 'bag' in tokens)
    # shoes
    shoe_word = any(w in tokens for w in ['shoe','shoes','sneaker','sneakers','boots'])
    shoes_color = 'white' if ('white' in tokens and shoe_word) else ('black' if ('black' in tokens and shoe_word) else ('other' if shoe_word else 'unknown'))
    # sleeve
    sleeve = 'long-sleeve' if ('long' in tokens and 'sleeve' in tokens) else ('short-sleeve' if ('short' in tokens and 'sleeve' in tokens) else ('sleeveless' if 'sleeveless' in tokens else 'unknown'))
    # hair
    hair = 'long hair' if ('long' in tokens and 'hair' in tokens) else ('short hair' if ('short' in tokens and 'hair' in tokens) else 'unknown')
    return dict(top_color=top_col, bottom_color=bot_col, bottom_type=bot_type, bag_present=bag_present, bag_absent=bag_absent, shoes_color=shoes_color, sleeve=sleeve, hair=hair)

def build_attr_sentences_v3(cap_list):
    base=cap_list[0] if cap_list else 'a person'
    a=parse_basic_attrs(base)
    s=[]
    if a['top_color']!='unknown': s.append(f"a person wearing a {a['top_color']} top")
    if a['bottom_type']!='unknown':
        bc=a['bottom_color'] if a['bottom_color']!='unknown' else ''
        if a['bottom_type']=='skirt': s.append(f"a person wearing {bc+' ' if bc else ''}skirt")
        else: s.append(f"a person wearing {bc+' ' if bc else ''}pants")
    if a['bag_present']: s.append("a person with a backpack")
    elif a['bag_absent']: s.append("no bag")
    if a['shoes_color'] in ['white','black','other']: s.append(f"a person wearing {a['shoes_color']} shoes")
    if a['sleeve']!='unknown': s.append(f"a person wearing {a['sleeve']}")
    if a['hair']!='unknown': s.append(f"a person with {a['hair']}")
    # ensure 5-8 sentences max
    return s[:8]

def pos_per_query(q_ids, g_ids):
    G=np.array(g_ids); return [int(np.sum(G==qid)) for qid in q_ids]

def filtered_indices(pos_counts):
    return [i for i,c in enumerate(pos_counts) if c>0]

def rerank_kr_jaccard_v3(S, I_base, topK=50, k1=20, k2=0, lam=0.3):
    Q,G=S.shape
    Dqg = 1 - S
    GG = 1 - (I_base@I_base.T)
    final = Dqg.copy()
    for i in range(Q):
        order=np.argsort(Dqg[i])[:min(topK,G)]
        Rq=list(order[:min(k1,len(order))])
        if k2>0:
            exp=set(Rq)
            for j in Rq:
                nbrs=np.argsort(GG[j])[:min(k2,G)]
                for nb in nbrs: exp.add(nb)
            Rq=list(exp)
        Rq=set(Rq)
        for j in order:
            Rj=list(np.argsort(GG[j])[:min(k1,G)])
            if k2>0:
                exp=set(Rj)
                for u in Rj:
                    nbrs=np.argsort(GG[u])[:min(k2,G)]
                    for nb in nbrs: exp.add(nb)
                Rj=list(exp)
            Rj=set(Rj)
            inter=len(Rq & Rj); uni=len(Rq | Rj)
            jac = 1 - (inter/uni if uni>0 else 0.)
            final[i,j] = lam*Dqg[i,j] + (1-lam)*jac
    return final

def kfold_logreg_fuse(S_list, q_ids, g_ids, out_dir):
    Zs=[]; stats=[]
    for S in S_list:
        Z,m,s = global_zscore(S); Zs.append(Z); stats.append({'mean':m,'std':s})
    X = np.stack([Z.flatten() for Z in Zs], axis=1)
    G=np.array(g_ids)
    y=[]
    for qid in q_ids: y.extend((G==qid).astype(np.int32).tolist())
    y=np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42) if SKLEARN_OK else None
    coefs=[]; intercepts=[]; yhat_all=np.zeros_like(y,dtype=float)
    idx_all=np.arange(len(y))
    if SKLEARN_OK:
        for train_idx, val_idx in kf.split(X):
            lr=LogisticRegression(max_iter=2000,class_weight='balanced')
            lr.fit(X[train_idx], y[train_idx])
            coefs.append(lr.coef_[0].astype(float)); intercepts.append(float(lr.intercept_[0]))
            yhat = lr.predict_proba(X[val_idx])[:,1]
            # Platt scaling on val
            pl_lr=LogisticRegression(max_iter=500)
            pl_lr.fit(yhat.reshape(-1,1), y[val_idx])
            yhat_cal = pl_lr.predict_proba(yhat.reshape(-1,1))[:,1]
            yhat_all[val_idx] = yhat_cal
        coef = np.mean(np.stack(coefs,axis=0),axis=0); interc=float(np.mean(intercepts))
        model_dump={'type':'sklearn_logreg_kfold','coef':coef.tolist(),'intercept':interc,'z_stats':stats}
        with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
        write_json(Path(out_dir,'calibration.json'), {'method':'platt','folds':5})
    else:
        # fallback single-pass manual logistic
        w=np.zeros(X.shape[1]); b=0.; lr_rate=0.05
        for _ in range(600):
            z=X@w+b; p=1/(1+np.exp(-z)); w-=lr_rate*(X.T@(p-y)/len(y)); b-=lr_rate*(np.sum(p-y)/len(y))
        coef=w.astype(float); interc=float(b)
        model_dump={'type':'manual_logreg','coef':coef.tolist(),'intercept':interc,'z_stats':stats}
        with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
        write_json(Path(out_dir,'calibration.json'), {'method':'none','folds':1})
        # use raw prob
        yhat_all = 1/(1+np.exp(-(X@w+b)))
    S_lr = yhat_all.reshape(S_list[0].shape)
    return S_lr

def main():
    ts_tag = ts()
    out_root = ROOT/'outputs/comparison'
    attr_dir = out_root/f"{ts_tag}_attrbank_v3"; ensure_dir(attr_dir)
    lr_dir = out_root/f"{ts_tag}_attrbank_v3_lr"; ensure_dir(lr_dir)
    rerank_dir = out_root/f"{ts_tag}_attrbank_v3_rerank"; ensure_dir(rerank_dir)
    overview_path = out_root/f"{ts_tag}_overview.csv"

    # run log
    for d in [attr_dir, lr_dir, rerank_dir]: Path(d/'run.log').write_text('',encoding='utf-8')

    # check inputs; do not stop, record errors
    errors=[]
    if not EXPL_CAP_PATH.exists(): errors.append(str(EXPL_CAP_PATH))
    if not LABELS_PATH.exists(): errors.append(str(LABELS_PATH))

    # data
    q_paths, g_paths = get_qg(); q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    pos_counts = pos_per_query(q_ids, g_ids)
    write_csv(attr_dir/'pos_per_query.csv', [[basename_noext(q_paths[i]), pos_counts[i]] for i in range(len(q_paths))], header=['query','pos_in_gallery'])

    # model and gallery embeds
    model, preprocess = load_clip()
    if EMB_IMAGE_PATH.exists():
        I_base = np.load(EMB_IMAGE_PATH)
        if I_base.shape[0]!=len(g_paths):
            I_base = encode_images(model, preprocess, g_paths); np.save(EMB_IMAGE_PATH, I_base)
    else:
        ensure_dir(EMB_IMAGE_PATH.parent)
        I_base = encode_images(model, preprocess, g_paths); np.save(EMB_IMAGE_PATH, I_base)
    # img self-sim stats
    GG = I_base@I_base.T
    write_json(attr_dir/'img_selfsim_stats.json', {'gg_mean':float(GG.mean()),'gg_std':float(GG.std())})

    # texts
    expl_caps = load_exp_caps(q_paths)
    expl_texts = [expl_caps[p][0] for p in q_paths]
    T_exp = encode_texts(model, expl_texts)
    S_exp = T_exp @ I_base.T

    attr_texts_per_q = [build_attr_sentences_v3(expl_caps[p]) for p in q_paths]
    S_attr_mean_rows=[]; S_attr_max_rows=[]
    for sents in attr_texts_per_q:
        if len(sents)==0:
            S_attr_mean_rows.append(np.zeros(len(g_paths),dtype=float))
            S_attr_max_rows.append(np.zeros(len(g_paths),dtype=float))
            continue
        V=encode_texts(model, sents)
        S_tmp = V @ I_base.T
        S_attr_mean_rows.append(np.mean(S_tmp, axis=0))
        S_attr_max_rows.append(np.max(S_tmp, axis=0))
    S_attr_mean = np.vstack(S_attr_mean_rows)
    S_attr_max = np.vstack(S_attr_max_rows)

    np.save(attr_dir/'S_exp.npy', S_exp)
    np.save(attr_dir/'S_attr_mean.npy', S_attr_mean)
    np.save(attr_dir/'S_attr_max.npy', S_attr_max)

    # metrics
    met_exp = rank1_map_ndcg(S_exp, q_ids, g_ids)
    met_mean = rank1_map_ndcg(S_attr_mean, q_ids, g_ids)
    met_max = rank1_map_ndcg(S_attr_max, q_ids, g_ids)
    write_json(attr_dir/'metrics.json', {'exp':met_exp,'attr_mean':met_mean,'attr_max':met_max})
    idx_keep = filtered_indices(pos_counts)
    filt = {
        'exp': rank1_map_ndcg(S_exp[idx_keep], [q_ids[i] for i in idx_keep], g_ids),
        'attr_mean': rank1_map_ndcg(S_attr_mean[idx_keep], [q_ids[i] for i in idx_keep], g_ids),
        'attr_max': rank1_map_ndcg(S_attr_max[idx_keep], [q_ids[i] for i in idx_keep], g_ids)
    }
    write_json(attr_dir/'filtered_metrics.json', filt)
    # summary (two rows)
    write_csv(attr_dir/'summary.csv', [
        ['attrbank_v3_mean', filt['attr_mean']['Rank-1'], filt['attr_mean']['mAP'], filt['attr_mean']['nDCG@10'], 'none','none'],
        ['attrbank_v3_max', filt['attr_max']['Rank-1'], filt['attr_max']['mAP'], filt['attr_max']['nDCG@10'], 'none','none']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])

    # config
    write_json(attr_dir/'config.yaml', {
        'model': MODEL_NAME,
        'zscore': 'global',
        'colors': COLORS_11,
        'explicit_path': str(EXPL_CAP_PATH),
        'filelists': {'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},
        'errors': errors
    })
    if errors: Path(attr_dir/'ERRORS.md').write_text("Missing inputs: \n"+"\n".join(errors), encoding='utf-8')

    # LR fusion on [z(S_exp), z(S_attr_mean), z(S_attr_max)]
    S_lr = kfold_logreg_fuse([S_exp, S_attr_mean, S_attr_max], q_ids, g_ids, lr_dir)
    np.save(lr_dir/'S_fuse_lr.npy', S_lr)
    met_lr = rank1_map_ndcg(S_lr, q_ids, g_ids)
    write_json(lr_dir/'metrics_fuse_lr.json', met_lr)
    filt_lr = rank1_map_ndcg(S_lr[idx_keep], [q_ids[i] for i in idx_keep], g_ids)
    write_csv(lr_dir/'summary.csv', [
        ['attrbank_v3_lr', filt_lr['Rank-1'], filt_lr['mAP'], filt_lr['nDCG@10'], 'lr','none']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    write_json(lr_dir/'config.yaml', {'fusion':'logreg_5fold_balanced','calibration':'platt'})

    # re-ranking grid scan on S_fuse_lr (Top-50)
    grid_k1=[10,20,30]; grid_k2=[0,6]; grid_l=[0.1,0.3]
    best={'score':-1,'k1':None,'k2':None,'l':None}
    for k1 in grid_k1:
        for k2 in grid_k2:
            for lam in grid_l:
                D = rerank_kr_jaccard_v3(S_lr, I_base, topK=min(50,len(g_paths)), k1=k1, k2=k2, lam=lam)
                np.save(rerank_dir/f"dist_k1{k1}_k2{k2}_l{lam}.npy", D)
                S_rr = 1 - D
                met = rank1_map_ndcg(S_rr, q_ids, g_ids)
                write_json(rerank_dir/f"metrics_k1{k1}_k2{k2}_l{lam}.json", met)
                score = met['nDCG@10']
                if score>best['score']:
                    best={'score':score,'k1':k1,'k2':k2,'l':lam,'met':met}
    # save best
    D_best = rerank_kr_jaccard_v3(S_lr, I_base, topK=min(50,len(g_paths)), k1=best['k1'], k2=best['k2'], lam=best['l'])
    np.save(rerank_dir/'dist.npy', D_best)
    S_best = 1 - D_best
    met_best = rank1_map_ndcg(S_best, q_ids, g_ids)
    write_json(rerank_dir/'metrics_rerank.json', met_best)
    write_csv(rerank_dir/'summary.csv', [
        ['attrbank_v3_rerank_best', met_best['Rank-1'], met_best['mAP'], met_best['nDCG@10'], 'lr','kr']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    write_json(rerank_dir/'config.yaml', {'grid':{'k1':grid_k1,'k2':grid_k2,'lambda':grid_l}, 'best': {'k1':best['k1'],'k2':best['k2'],'lambda':best['l']}})

    # README (<=200 chars), pass/fail
    def pass_line(cur, base): return cur['mAP']>base['mAP'] or cur['nDCG@10']>base['nDCG@10']
    # try to load previous baseline
    prev_attr_dirs = sorted([p for p in (ROOT/'outputs/comparison').glob('*_attrbank*') if 'v3' not in p.name])
    base_met={'Rank-1':0.,'mAP':0.,'nDCG@10':0.}
    if prev_attr_dirs:
        base_dir = prev_attr_dirs[-1]
        try:
            rows=Path(base_dir/'summary.csv').read_text(encoding='utf-8').splitlines()
            if len(rows)>=2:
                parts=rows[1].split(',')
                base_met={'Rank-1':float(parts[1]),'mAP':float(parts[2]),'nDCG@10':float(parts[3])}
        except: pass
    # choose v3 best of mean/max for overview entry
    use_v3 = met_mean if met_mean['nDCG@10']>=met_max['nDCG@10'] else met_max
    Path(attr_dir/'README.md').write_text(f"短句属性银行v3：生成5-8条短句，mean/max聚合。nDCG@10={use_v3['nDCG@10']}, 是否过线：{pass_line(use_v3, base_met)}", encoding='utf-8')
    Path(lr_dir/'README.md').write_text(f"LogReg融合（5-fold+Platt），nDCG@10={met_lr['nDCG@10']}", encoding='utf-8')
    Path(rerank_dir/'README.md').write_text(f"k-reciprocal+Jaccard网格扫描Top-50，best(k1={best['k1']},k2={best['k2']},l={best['l']})，nDCG@10={met_best['nDCG@10']}", encoding='utf-8')

    # overview: add four rows
    header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank','best_k1,k2,lambda']
    rows=[
        ['attrbank_v3', use_v3['Rank-1'], use_v3['mAP'], use_v3['nDCG@10'], 'none','none',''],
        ['attrbank_v3_lr', met_lr['Rank-1'], met_lr['mAP'], met_lr['nDCG@10'], 'lr','none',''],
        ['attrbank_v3_rerank_best', met_best['Rank-1'], met_best['mAP'], met_best['nDCG@10'], 'lr','kr', f"{best['k1']},{best['k2']},{best['l']}"],
        ['attrbank_prev', base_met['Rank-1'], base_met['mAP'], base_met['nDCG@10'], 'none','none','']
    ]
    write_csv(overview_path, rows, header=header)

if __name__=='__main__':
    main()