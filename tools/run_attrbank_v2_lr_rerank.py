import os, json, time, pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import clip

# Try sklearn, fallback to manual LR if unavailable
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK=True
except Exception:
    SKLEARN_OK=False

ROOT = Path('d:/PRP SunnyLab/ReID')
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
QUERY_LIST = ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
SEED=42
np.random.seed(SEED)

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
        if len(pos)==0: ap=0.
        else:
            prec=[np.sum(rel[:idx+1])/(idx+1) for idx in pos]
            ap=float(np.mean(prec))
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

# --- AttrBank v2 ---
COLORS=['black','white','blue','red','green','yellow','purple','pink','orange','brown','gray']
TOP_TYPES=['t-shirt','shirt','blouse','hoodie','sweater','jacket','coat','vest','polo']
BOT_TYPES=['pants','trousers','jeans','shorts','skirt','leggings']
SLEEVE=['long-sleeve','short-sleeve','sleeveless']

def parse_attrs(caption):
    t=caption.lower()
    tokens=t.replace(',',' ').replace('.',' ').split()
    def find_first(words, tokens):
        for w in words:
            if w in tokens: return w
        return 'unknown'
    top_color=find_first(COLORS,tokens)
    bot_color=find_first(COLORS,tokens)
    top_type=find_first(TOP_TYPES,tokens)
    bot_type=find_first(BOT_TYPES,tokens)
    if 'long sleeve' in t or 'long-sleeve' in t: sl='long-sleeve'
    elif 'short sleeve' in t or 'short-sleeve' in t: sl='short-sleeve'
    elif 'sleeveless' in t: sl='sleeveless'
    else: sl='unknown'
    shoes='white' if ('white' in tokens and 'shoe' in t) or 'white shoes' in t else ('black' if ('black' in tokens and 'shoe' in t) or 'black shoes' in t else 'other')
    bag = 'with bag' if any(k in t for k in ['bag','backpack','handbag','shoulder bag']) else 'no bag'
    return dict(top_color=top_color,bot_color=bot_color,top_type=top_type,bot_type=bot_type,sleeve=sl,shoes=shoes,bag=bag)

def build_attr_sentences_v2(cap_list):
    base=cap_list[0] if cap_list else 'a person'
    a=parse_attrs(base)
    def part_top():
        if a['top_type']!='unknown' and a['top_color']!='unknown': return f"{a['top_color']} {a['top_type']}"
        if a['top_type']!='unknown': return a['top_type']
        if a['top_color']!='unknown': return f"{a['top_color']} top"
        return None
    def part_bot():
        if a['bot_type']!='unknown' and a['bot_color']!='unknown': return f"{a['bot_color']} {a['bot_type']}"
        if a['bot_type']!='unknown': return a['bot_type']
        if a['bot_color']!='unknown': return f"{a['bot_color']} bottoms"
        return None
    parts=[]
    pt=part_top(); pb=part_bot()
    if pt: parts.append(pt)
    if pb: parts.append(pb)
    bag=a['bag']
    sleeve = a['sleeve'] if a['sleeve']!='unknown' else None
    shoes = a['shoes']
    s=[]
    # Compose sentences focusing on known attributes only
    if parts:
        s.append("a person wearing "+", ".join(parts))
    else:
        s.append("a person")
    if sleeve:
        s.append(f"a pedestrian, {sleeve}")
    s.append(f"person, {bag}")
    if shoes!='other':
        s.append(f"human figure with {shoes} shoes")
    if pt:
        s.append(f"a person in the scene wearing {pt}")
    if pb:
        s.append(f"a person standing, {pb}")
    return s

def pos_per_query(q_ids, g_ids):
    G=np.array(g_ids); return [int(np.sum(G==qid)) for qid in q_ids]

def filtered_indices(pos_counts):
    return [i for i,c in enumerate(pos_counts) if c>0]

def fit_lr_fuse(S_exp, S_attr, q_ids, g_ids, out_dir):
    # global zscore features
    Zexp, m1, s1 = global_zscore(S_exp)
    Zattr, m2, s2 = global_zscore(S_attr)
    X = np.stack([Zexp.flatten(), Zattr.flatten()], axis=1)
    y=[]
    G=np.array(g_ids)
    for i,qid in enumerate(q_ids):
        rel=(G==qid).astype(np.int32).tolist(); y.extend(rel)
    y=np.array(y)
    if SKLEARN_OK:
        lr=LogisticRegression(max_iter=1000,class_weight='balanced')
        lr.fit(X,y)
        w=lr.coef_[0].astype(float); b=float(lr.intercept_[0])
        yhat=lr.predict_proba(X)[:,1]
        model_dump={'type':'sklearn_logreg','coef':w.tolist(),'intercept':b,'z1_mean':m1,'z1_std':s1,'z2_mean':m2,'z2_std':s2}
        with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
    else:
        # manual GD
        w=np.zeros(2); b=0.; lr_rate=0.05
        for _ in range(300):
            z=X@w + b; p=1/(1+np.exp(-z)); 
            grad_w = X.T@(p-y)/len(y); grad_b = np.sum(p-y)/len(y)
            w -= lr_rate*grad_w; b -= lr_rate*grad_b
        yhat=1/(1+np.exp(-(X@w+b)))
        model_dump={'type':'manual_logreg','coef':w.tolist(),'intercept':float(b),'z1_mean':m1,'z1_std':s1,'z2_mean':m2,'z2_std':s2}
        with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
    S_lr = yhat.reshape(S_exp.shape)
    write_json(Path(out_dir,'metrics_fuse_lr.json'), rank1_map_ndcg(S_lr, q_ids, g_ids))
    return S_lr

def rerank_kr_jaccard(S, I_base, topK=100, k1=20, lam=0.3):
    # S is similarity QxG; convert to distance
    Q,G=S.shape
    Dqg = 1 - S
    # gallery-gallery distance from embeddings
    I=I_base
    GG = 1 - (I@I.T)  # cosine distance
    final = Dqg.copy()
    for i in range(Q):
        order=np.argsort(Dqg[i])[:min(topK,G)]
        Rq=set(order[:min(k1,len(order))])
        for j in order:
            Rj=set(np.argsort(GG[j])[:min(k1,G)])
            inter=len(Rq & Rj); uni=len(Rq | Rj)
            jac = 1 - (inter/uni if uni>0 else 0.)
            final[i,j] = lam*Dqg[i,j] + (1-lam)*jac
    return final

def main():
    ts_tag = ts()
    out_root = ROOT/'outputs/comparison'
    attr_dir = out_root/f"{ts_tag}_attrbank_v2"; ensure_dir(attr_dir)
    rerank_dir = out_root/f"{ts_tag}_attrbank_v2_rerank"; ensure_dir(rerank_dir)
    imp_dir = out_root/f"{ts_tag}_imp_ens"; ensure_dir(imp_dir)

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

    # texts: explicit + attrbank v2
    expl_caps = load_exp_caps(q_paths)
    expl_texts = [expl_caps[p][0] for p in q_paths]
    T_exp = encode_texts(model, expl_texts)
    attr_texts_per_q = [build_attr_sentences_v2(expl_caps[p]) for p in q_paths]
    # max-over-sentences aggregation at similarity level (improves recall)
    S_attr_rows=[]
    for sents in attr_texts_per_q:
        V=encode_texts(model, sents)  # n_sent x d
        S_tmp = V @ I_base.T          # n_sent x G
        S_attr_rows.append(np.max(S_tmp, axis=0))
    S_attr = np.vstack(S_attr_rows)

    # sims
    S_exp = T_exp @ I_base.T

    # LR fusion
    S_lr = fit_lr_fuse(S_exp, S_attr, q_ids, g_ids, attr_dir)
    write_json(attr_dir/'metrics_exp.json', rank1_map_ndcg(S_exp, q_ids, g_ids))
    write_json(attr_dir/'metrics_attr.json', rank1_map_ndcg(S_attr, q_ids, g_ids))
    # filtered metrics
    idx_keep = filtered_indices(pos_counts)
    filt = {
        'exp': rank1_map_ndcg(S_exp[idx_keep], [q_ids[i] for i in idx_keep], g_ids),
        'attr': rank1_map_ndcg(S_attr[idx_keep], [q_ids[i] for i in idx_keep], g_ids),
        'lr_fuse': rank1_map_ndcg(S_lr[idx_keep], [q_ids[i] for i in idx_keep], g_ids)
    }
    write_json(attr_dir/'filtered_metrics.json', filt)
    # summary
    write_csv(attr_dir/'summary.csv', [
        ['attrbank_v2', filt['attr']['Rank-1'], filt['attr']['mAP'], filt['attr']['nDCG@10'], 'none','none'],
        ['attrbank_v2_lr', filt['lr_fuse']['Rank-1'], filt['lr_fuse']['mAP'], filt['lr_fuse']['nDCG@10'], 'lr','none']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])

    # rerank
    D_final = rerank_kr_jaccard(S_lr, I_base, topK=min(100,len(g_paths)), k1=10, lam=0.1)
    np.save(rerank_dir/'dist.npy', D_final)
    S_rr = 1 - D_final
    met_rr = rank1_map_ndcg(S_rr, q_ids, g_ids)
    write_json(rerank_dir/'metrics_rerank.json', met_rr)
    write_json(rerank_dir/'filtered_metrics.json', rank1_map_ndcg(S_rr[idx_keep], [q_ids[i] for i in idx_keep], g_ids))
    write_csv(rerank_dir/'summary.csv', [
        ['attrbank_v2_rerank', met_rr['Rank-1'], met_rr['mAP'], met_rr['nDCG@10'], 'lr','kr']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])

    # implicit ensemble
    templates=[
        'a person','a pedestrian','a person walking','a person standing','a person in the scene','a human figure'
    ]
    T_imps = encode_texts(model, templates)  # 6 x d
    # For each query, use six templates; similarity via max across templates
    S_imp_list=[]
    for _ in q_paths:
        S_tmp = T_imps @ I_base.T  # 6 x G
        S_imp_list.append(np.max(S_tmp, axis=0))
    S_imp_ens = np.stack(S_imp_list, axis=0)  # Q x G
    np.save(imp_dir/'S_imp_ens.npy', S_imp_ens)
    met_imp = rank1_map_ndcg(S_imp_ens, q_ids, g_ids)
    write_json(imp_dir/'metrics_imp_ens.json', met_imp)
    # LR fusion with explicit as reference
    Zexp,_m1,_s1 = global_zscore(S_exp)
    Zimp,_m2,_s2 = global_zscore(S_imp_ens)
    X=np.stack([Zexp.flatten(), Zimp.flatten()], axis=1)
    y=[]; G=np.array(g_ids)
    for qid in q_ids: y.extend((G==qid).astype(np.int32).tolist())
    if SKLEARN_OK:
        lr=LogisticRegression(max_iter=1000,class_weight='balanced'); lr.fit(X,y); yhat=lr.predict_proba(X)[:,1]
    else:
        w=np.zeros(2); b=0.; lr_rate=0.05
        for _ in range(300): z=X@w+b; p=1/(1+np.exp(-z)); w-=lr_rate*(X.T@(p-y)/len(y)); b-=lr_rate*(np.sum(p-y)/len(y))
        yhat=1/(1+np.exp(-(X@w+b)))
    S_imp_lr = yhat.reshape(S_exp.shape)
    write_json(imp_dir/'metrics_imp_ens_lr_fuse.json', rank1_map_ndcg(S_imp_lr, q_ids, g_ids))
    write_json(imp_dir/'filtered_metrics.json', rank1_map_ndcg(S_imp_lr[idx_keep], [q_ids[i] for i in idx_keep], g_ids))

    # overview
    ov=[
        ['attrbank_v2', filt['attr']['Rank-1'], filt['attr']['mAP'], filt['attr']['nDCG@10'], 'none','none'],
        ['attrbank_v2_lr', filt['lr_fuse']['Rank-1'], filt['lr_fuse']['mAP'], filt['lr_fuse']['nDCG@10'], 'lr','none'],
        ['attrbank_v2_rerank', met_rr['Rank-1'], met_rr['mAP'], met_rr['nDCG@10'], 'lr','kr'],
        ['imp_ens', met_imp['Rank-1'], met_imp['mAP'], met_imp['nDCG@10'], 'none','none']
    ]
    write_csv(out_root/f"{ts_tag}_overview.csv", ov, header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])

if __name__=='__main__':
    main()