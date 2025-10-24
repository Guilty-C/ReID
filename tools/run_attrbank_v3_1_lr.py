import os, json, time, pickle
from pathlib import Path
import numpy as np
import torch, clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
# prefer subset100 lists if available
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

# --- utils ---
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

def get_qg(): q=read_lines(QUERY_LIST); g=read_lines(GALLERY_LIST); return q,g

def ids_from_paths(paths):
    ids=[]
    for p in paths:
        s=basename_noext(p); head=s.split('_')[0]
        try: ids.append(int(head))
        except: ids.append(head)
    return ids

def pos_per_query(q_ids,g_ids):
    G=np.array(g_ids); return [int(np.sum(G==qid)) for qid in q_ids]

def filtered_indices(pos_counts): return [i for i,c in enumerate(pos_counts) if c>0]

def global_zscore(S):
    m=float(S.mean()); s=float(S.std()) if S.std()>1e-9 else 1.0
    return (S-m)/s, m, s

def rank1_map_ndcg(S, q_ids, g_ids, k_ndcg=10):
    G=np.array(g_ids); n_q=S.shape[0]; r1=0; ap_sum=0; nd_sum=0
    for i in range(n_q):
        rel=(G==q_ids[i]).astype(np.int32)
        order=np.argsort(-S[i])
        r1+=int(rel[order[0]]==1)
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

# --- CLIP ---

def load_clip():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return model, preprocess

def encode_images(model, preprocess, paths):
    from PIL import Image
    embs=[]
    for p in paths:
        try:
            img=Image.open(p).convert('RGB')
            x=preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): v=model.encode_image(x)
            v=v/ v.norm(dim=-1, keepdim=True)
            embs.append(v.squeeze(0).cpu().numpy())
        except Exception:
            embs.append(np.zeros(model.visual.output_dim, dtype=np.float32))
    return np.stack(embs,axis=0)

def encode_texts(model, texts):
    tokens=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(tokens)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

# --- explicit & attrs (v3.1) ---

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
    for c in COLORS_11:
        if c in t: return c
    # fallback via variants tokens
    for c,vars in COLOR_VARIANTS.items():
        for v in vars:
            if v in t: return c
    return 'unknown'

def parse_basic_colors_bottom(caption):
    t=caption.lower(); tokens=t.replace(',',' ').replace('.',' ').split()
    # detect explicit tokens or variants
    top_color=normalize_color(t)
    bot_type='skirt' if 'skirt' in tokens else ('pants' if any(w in tokens for w in ['pant','pants','trouser','trousers','jeans','shorts','leggings']) else 'unknown')
    bot_color=normalize_color(t)
    return dict(top_color=top_color,bottom_color=bot_color,bottom_type=bot_type)

def build_attr_sentences_v3_1(cap_list):
    base=cap_list[0] if cap_list else 'a person'
    a=parse_basic_colors_bottom(base)
    s=[]
    if a['top_color']!='unknown':
        for v in COLOR_VARIANTS[a['top_color']]: s.append(f"a person wearing a {v} top")
    if a['bottom_type']!='unknown' and a['bottom_color']!='unknown':
        for v in COLOR_VARIANTS[a['bottom_color']]: s.append(f"a person wearing {v} {a['bottom_type']}")
    return s[:16]

# --- LR fusion (robust) ---

def kfold_logreg_fuse_robust(S_list, q_ids, g_ids, out_dir):
    Zs=[]; stats=[]
    for S in S_list:
        Z,m,s=global_zscore(S); Zs.append(Z); stats.append({'mean':m,'std':s})
    X=np.stack([Z.flatten() for Z in Zs], axis=1)
    G=np.array(g_ids); y=[]
    for qid in q_ids: y.extend((G==qid).astype(np.int32).tolist())
    y=np.array(y)
    pos_cnt=int(np.sum(y==1))
    if pos_cnt<2:
        # fallback global LR
        lr=LogisticRegression(max_iter=2000,class_weight='balanced'); lr.fit(X,y)
        yhat=lr.predict_proba(X)[:,1]
        model_dump={'type':'sklearn_logreg_global','coef':lr.coef_[0].astype(float).tolist(),'intercept':float(lr.intercept_[0]),'z_stats':stats,'pos_cnt':pos_cnt}
        with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
        write_json(Path(out_dir,'calibration.json'), {'method':'none','folds':1,'fallback':'global'})
        return yhat.reshape(S_list[0].shape)
    n_splits=max(2, min(5, pos_cnt))
    skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    yhat_all=np.zeros_like(y,dtype=float); coefs=[]; intercepts=[]; folds=0
    for tr,va in skf.split(X,y):
        folds+=1
        if len(np.unique(y[tr]))<2:
            lr=LogisticRegression(max_iter=2000,class_weight='balanced'); lr.fit(X,y)
            yhat=yhat_all.copy();
            yhat[va]=lr.predict_proba(X[va])[:,1]
            coefs.append(lr.coef_[0].astype(float)); intercepts.append(float(lr.intercept_[0]))
        else:
            lr=LogisticRegression(max_iter=2000,class_weight='balanced'); lr.fit(X[tr],y[tr])
            yh=lr.predict_proba(X[va])[:,1]
            # Platt scaling if both classes in val
            if len(np.unique(y[va]))>=2:
                pl=LogisticRegression(max_iter=500); pl.fit(yh.reshape(-1,1), y[va]); yh=pl.predict_proba(yh.reshape(-1,1))[:,1]
            yhat_all[va]=yh
            coefs.append(lr.coef_[0].astype(float)); intercepts.append(float(lr.intercept_[0]))
    coef=np.mean(np.stack(coefs,axis=0),axis=0); interc=float(np.mean(intercepts))
    model_dump={'type':'sklearn_logreg_kfold','coef':coef.tolist(),'intercept':interc,'z_stats':stats,'folds':folds,'pos_cnt':pos_cnt}
    with open(Path(out_dir,'weights.pkl'),'wb') as f: pickle.dump(model_dump,f)
    write_json(Path(out_dir,'calibration.json'), {'method':'platt_if_possible','folds':folds})
    return yhat_all.reshape(S_list[0].shape)

# --- main ---

def main():
    ts_tag=ts(); attr_dir=OUT_ROOT/f"{ts_tag}_attrbank_v3_1"; lr_dir=OUT_ROOT/f"{ts_tag}_attrbank_v3_1_lr"; overview_path=OUT_ROOT/f"{ts_tag}_overview.csv"
    for d in [attr_dir, lr_dir]: ensure_dir(d); Path(d/'run.log').write_text('',encoding='utf-8')
    # data
    q_paths, g_paths = get_qg(); q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    pos_counts=pos_per_query(q_ids,g_ids); write_csv(attr_dir/'pos_per_query.csv', [[basename_noext(q_paths[i]), pos_counts[i]] for i in range(len(q_paths))], header=['query','pos_in_gallery'])
    # model and gallery embeds
    model, preprocess = load_clip()
    if EMB_IMAGE_PATH.exists():
        I_base=np.load(EMB_IMAGE_PATH)
        if I_base.shape[0]!=len(g_paths): I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    else:
        ensure_dir(EMB_IMAGE_PATH.parent); I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    # texts: explicit (canonical color only) + v3.1 attrs (color variants, max-over-sentences per-attribute)
    expl_caps=load_exp_caps(q_paths)
    # explicit: use canonical top/bottom sentences once each (no variants), mean aggregation
    S_exp_rows=[]; S_attr_mean_rows=[]; S_attr_max_rows=[]
    for p in q_paths:
        caps=expl_caps[p]; sents_attr=build_attr_sentences_v3_1(caps)
        # explicit canonical (first variant only if available)
        sents_exp=[]
        if sents_attr:
            # pick canonical two if present
            tops=[s for s in sents_attr if ' top' in s]; bots=[s for s in sents_attr if ' pants' in s or ' skirt' in s]
            if tops: sents_exp.append(tops[0])
            if bots: sents_exp.append(bots[0])
        else:
            sents_exp.append('a person')
        V_exp=encode_texts(model, sents_exp); S_exp_rows.append(np.mean(V_exp@I_base.T, axis=0))
        if len(sents_attr)==0:
            S_attr_mean_rows.append(np.zeros(len(g_paths))); S_attr_max_rows.append(np.zeros(len(g_paths)))
        else:
            V_attr=encode_texts(model, sents_attr); S_tmp=V_attr@I_base.T
            # group-level max-over-sentences: top group and bottom group
            tops=[i for i,s in enumerate(sents_attr) if ' top' in s]; bots=[i for i,s in enumerate(sents_attr) if ' pants' in s or ' skirt' in s]
            top_sc = np.max(S_tmp[tops], axis=0) if tops else np.zeros(len(g_paths))
            bot_sc = np.max(S_tmp[bots], axis=0) if bots else np.zeros(len(g_paths))
            S_attr_mean_rows.append((top_sc+bot_sc)/ ( (1 if tops else 0)+(1 if bots else 0) if ((1 if tops else 0)+(1 if bots else 0))>0 else 1))
            S_attr_max_rows.append(np.maximum(top_sc, bot_sc))
    S_exp=np.vstack(S_exp_rows); S_attr_mean=np.vstack(S_attr_mean_rows); S_attr_max=np.vstack(S_attr_max_rows)
    np.save(attr_dir/'S_exp.npy', S_exp); np.save(attr_dir/'S_attr_mean.npy', S_attr_mean); np.save(attr_dir/'S_attr_max.npy', S_attr_max)
    # metrics
    met_exp=rank1_map_ndcg(S_exp,q_ids,g_ids); met_mean=rank1_map_ndcg(S_attr_mean,q_ids,g_ids); met_max=rank1_map_ndcg(S_attr_max,q_ids,g_ids)
    write_json(attr_dir/'metrics.json', {'exp':met_exp,'attr_mean':met_mean,'attr_max':met_max})
    idx_keep=filtered_indices(pos_counts)
    filt={'exp':rank1_map_ndcg(S_exp[idx_keep],[q_ids[i] for i in idx_keep],g_ids),
          'attr_mean':rank1_map_ndcg(S_attr_mean[idx_keep],[q_ids[i] for i in idx_keep],g_ids),
          'attr_max':rank1_map_ndcg(S_attr_max[idx_keep],[q_ids[i] for i in idx_keep],g_ids)}
    write_json(attr_dir/'filtered_metrics.json',filt)
    write_csv(attr_dir/'summary.csv', [
        ['attrbank_v3_1_mean', filt['attr_mean']['Rank-1'], filt['attr_mean']['mAP'], filt['attr_mean']['nDCG@10'], 'none','none'],
        ['attrbank_v3_1_max', filt['attr_max']['Rank-1'], filt['attr_max']['mAP'], filt['attr_max']['nDCG@10'], 'none','none']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    # config & README
    write_json(attr_dir/'config.yaml', {'model':MODEL_NAME,'zscore':'global','colors':COLORS_11,'variants':COLOR_VARIANTS,'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},'bag_shoes':'disabled','seed':SEED})
    Path(attr_dir/'README.md').write_text('v3.1颜色强化：仅top/bottom两句型，颜色11类+同义变体，属性组内max聚合；禁用包/鞋。', encoding='utf-8')
    # LR fusion on three features
    S_lr=kfold_logreg_fuse_robust([S_exp,S_attr_mean,S_attr_max], q_ids,g_ids, lr_dir)
    np.save(lr_dir/'S_fuse_lr.npy', S_lr)
    met_lr=rank1_map_ndcg(S_lr,q_ids,g_ids)
    write_json(lr_dir/'metrics_fuse_lr.json', met_lr)
    filt_lr=rank1_map_ndcg(S_lr[idx_keep],[q_ids[i] for i in idx_keep],g_ids)
    write_csv(lr_dir/'summary.csv', [['attrbank_v3_1_lr', filt_lr['Rank-1'], filt_lr['mAP'], filt_lr['nDCG@10'], 'lr','none']], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    write_json(lr_dir/'config.yaml', {'fusion':'logreg_stratified_kfold_balanced','calibration':'platt_if_possible','seed':SEED})
    Path(lr_dir/'README.md').write_text('v3.1_LR融合：输入三路(z-exp/z-mean/z-max)，StratifiedKFold+Platt，失败回退全局LR或跳过校准。', encoding='utf-8')
    # overview
    rows=[
        ['attrbank_v3_1_mean', filt['attr_mean']['Rank-1'], filt['attr_mean']['mAP'], filt['attr_mean']['nDCG@10'], 'none','none'],
        ['attrbank_v3_1_max', filt['attr_max']['Rank-1'], filt['attr_max']['mAP'], filt['attr_max']['nDCG@10'], 'none','none'],
        ['attrbank_v3_1_lr', filt_lr['Rank-1'], filt_lr['mAP'], filt_lr['nDCG@10'], 'lr','none']
    ]
    write_csv(overview_path, rows, header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])

if __name__=='__main__':
    main()