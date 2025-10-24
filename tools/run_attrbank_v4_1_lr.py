import os, json, time, pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
QUERY_LIST = ROOT/'larger_iso/64/query100.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt'
SEED=42
np.random.seed(SEED)

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
        s=basename_noext(p)
        head=s.split('_')[0]
        try: ids.append(int(head))
        except: ids.append(head)
    return ids

def global_zscore(S):
    m=float(S.mean()); s=float(S.std()) if S.std()>1e-9 else 1.0
    return (S-m)/s, m, s

# latest dirs

def latest_dir_suffix(suffix):
    cand=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith(suffix)]
    return sorted(cand)[-1] if cand else None

# load features

def load_exp_best():
    d=latest_dir_suffix('_attrbank_v4_1_triwin_grid_soft') or latest_dir_suffix('_attrbank_v4_1_triwin_grid')
    if not d: raise FileNotFoundError('StepB not found')
    cfg=json.loads((d/'config.yaml').read_text(encoding='utf-8'))
    key=cfg.get('best_exp_key') or 'w1'
    S=np.load(d/f'S_exp_tri_{key}.npy')
    return d, key, S


def load_attr_triwin():
    d=latest_dir_suffix('_attrbank_v4_triwin_weighted')
    if not d: raise FileNotFoundError('v4 triwin weighted not found')
    S_mean=np.load(d/'S_attr_tri_mean.npy')
    S_max=np.load(d/'S_attr_tri_max.npy')
    return d, S_mean, S_max


def load_consistency():
    d=latest_dir_suffix('_attrbank_v4_1_colorcons_soft') or latest_dir_suffix('_attrbank_v4_1_colorcons')
    if not d: raise FileNotFoundError('StepA-soft not found')
    C=np.load(d/'consistency.npy')
    # also load labels for mismatch later
    q_labels=json.loads((d/'query_color_labels.json').read_text(encoding='utf-8')) if (d/'query_color_labels.json').exists() else {}
    g_labels=json.loads((d/'gallery_color_labels.json').read_text(encoding='utf-8')) if (d/'gallery_color_labels.json').exists() else {}
    return d, C, q_labels, g_labels

# metrics

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

# top1 mismatch ratio among queries with known top/bottom both

def top1_mismatch_ratio(S, q_labels, g_labels):
    if not q_labels or not g_labels: return 0.0, 0
    n=S.shape[0]; mism=0; known=0
    for i in range(n):
        order=np.argsort(-S[i]); j=order[0]
        ql=q_labels.get(str(i)) or q_labels.get(basename_noext(read_lines(QUERY_LIST)[i])) or {}
        gl=g_labels.get(str(j)) or g_labels.get(basename_noext(read_lines(GALLERY_LIST)[j])) or {}
        qt=ql.get('top','unknown'); qb=ql.get('bottom','unknown')
        gt=gl.get('top','unknown'); gb=gl.get('bottom','unknown')
        if qt!='unknown' and qb!='unknown' and gt!='unknown' and gb!='unknown':
            known+=1
            if qt!=gt or qb!=gb: mism+=1
    return (mism/known if known>0 else 0.0), known


def known_coverage_rate(qlab):
    n=len(qlab);
    if n==0: return 0.0
    known_any=sum(1 for v in qlab.values() if v.get('top_color')!='unknown' or v.get('bottom_color')!='unknown')
    return known_any/float(n)

def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_lr_soft"; ensure_dir(out_dir)
    # data
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    # features
    tri_b_dir, best_key, S_exp_best = load_exp_best()
    tri_v4_dir, S_attr_mean, S_attr_max = load_attr_triwin()
    color_dir, C, qlab, glab = load_consistency()
    Z_exp,_m,_s = global_zscore(S_exp_best)
    Z_attr_mean,_m,_s = global_zscore(S_attr_mean)
    Z_attr_max,_m,_s = global_zscore(S_attr_max)
    # LR
    Q,G = Z_exp.shape
    X=np.stack([Z_exp.flatten(), Z_attr_mean.flatten(), Z_attr_max.flatten(), C.flatten()], axis=1)
    y=np.array([1 if g_ids[j]==q_ids[i] else 0 for i in range(Q) for j in range(G)])
    pos=int(np.sum(y==1))
    n_folds = int(min(5, max(1,pos)))
    clf = LogisticRegression(class_weight='balanced', max_iter=1500, solver='liblinear')
    weights=None; calib={'method':'none','folds':n_folds,'pos':pos}
    if n_folds>=2:
        skf=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        calibrated=CalibratedClassifierCV(clf, method='sigmoid', cv=skf)
        calibrated.fit(X,y)
        probs=calibrated.predict_proba(X)[:,1]
        weights=[]
        for cc in getattr(calibrated,'calibrated_classifiers_',[]):
            be=getattr(cc,'base_estimator',None)
            if be is not None and hasattr(be,'coef_'):
                weights.append({'coef':be.coef_.tolist(),'intercept':be.intercept_.tolist()})
        calib={'method':'platt','folds':n_folds,'pos':pos}
    else:
        clf.fit(X,y)
        probs=clf.predict_proba(X)[:,1]
        weights={'coef':clf.coef_.tolist(),'intercept':clf.intercept_.tolist()}
        calib={'method':'none','folds':1,'pos':pos}
    S_fuse=probs.reshape(Q,G)
    np.save(out_dir/'S_fuse_lr.npy', S_fuse)
    met=rank1_map_ndcg(S_fuse, q_ids, g_ids)
    write_json(out_dir/'metrics_fuse_lr.json', met)
    write_json(out_dir/'calibration.json', calib)
    with open(out_dir/'weights.pkl','wb') as f: pickle.dump(weights,f)
    # mismatch on final fused top1
    ratio, known = top1_mismatch_ratio(S_fuse, qlab, glab)
    # summary
    write_csv(out_dir/'summary.csv', [[ 'attrbank_v4_1_lr_soft', met['Rank-1'], met['mAP'], met['nDCG@10'], 'LR+consistency', 'none', round(ratio,4) ]], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank','top1_mismatch'])
    write_json(out_dir/'config.yaml', {
        'features':['z(S_exp_tri_w_exp)','z(S_attr_tri_mean_w_attr)','z(S_attr_tri_max_w_attr)','consistency'],
        'best_exp_key': best_key,
        'triwin_w_attr': 'w2',
        'folds': n_folds,
        'class_weight':'balanced',
        'seed': SEED,
        'dirs': {'triwin_grid_soft': str(tri_b_dir), 'triwin_v4': str(tri_v4_dir), 'colorcons_soft': str(color_dir)}
    })
    Path(out_dir/'README.md').write_text('v4.1-soft LR融合：输入四路(z-exp_best/z-attr_mean/z-attr_max/一致性)，StratifiedKFold+Platt；折单类回退全量LR。', encoding='utf-8')
    # overview append (soft schema)
    overview=OUT_ROOT/f"{ts_tag}_overview.csv"
    known_rate = known_coverage_rate(qlab)
    write_csv(overview, [[ 'attrbank_v4_1_lr_soft', '-', best_key, 'w2', round(known_rate,4), round(ratio,4) ]], header=['fusion','gamma','triwin_w_exp','triwin_w_attr','known_rate','top1_mismatch'])

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        out_dir=OUT_ROOT/f"{ts()}_attrbank_v4_1_lr_soft_error"; ensure_dir(out_dir)
        Path(out_dir/'ERRORS.md').write_text(f'Error in StepC: {repr(e)}', encoding='utf-8')