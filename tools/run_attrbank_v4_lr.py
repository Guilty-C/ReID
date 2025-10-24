import os, json, time, pickle
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
QUERY_LIST = ROOT/'larger_iso/64/query100.txt' if (ROOT/'larger_iso/64/query100.txt').exists() else ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt' if (ROOT/'larger_iso/64/gallery100.txt').exists() else ROOT/'larger_iso/64/gallery.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
SEED=42
np.random.seed(SEED)

COLORS_11=['black','white','gray','red','orange','yellow','green','blue','purple','pink','brown']

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
    return v.squeeze(0).cpu().numpy() if len(texts)==1 else v.cpu().numpy()

def center_square_crop(img):
    w,h=img.size; s=min(w,h); left=(w-s)//2; top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def top_bottom_crops(img):
    sq=center_square_crop(img)
    w,h=sq.size
    return sq.crop((0,0,w,h//2)), sq.crop((0,h//2,w,h))

def encode_gallery_window_embeddings(model, preprocess, g_paths):
    I_top=[]; I_bottom=[]
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
            I_bottom.append(enc(bot_img))
        except Exception:
            d=model.visual.output_dim
            I_top.append(np.zeros(d)); I_bottom.append(np.zeros(d))
    return np.stack(I_top), np.stack(I_bottom)

# latest triwin outputs

def latest_dir_with(prefix):
    dirs=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith(prefix)]
    return sorted(dirs)[-1] if dirs else None

def load_triwin_features():
    d=latest_dir_with('_attrbank_v4_triwin_weighted')
    if d is None:
        raise FileNotFoundError('triwin output not found')
    S_exp_tri=np.load(d/'S_exp_tri.npy')
    S_attr_tri_mean=np.load(d/'S_attr_tri_mean.npy')
    S_attr_tri_max=np.load(d/'S_attr_tri_max.npy')
    return d, S_exp_tri, S_attr_tri_mean, S_attr_tri_max

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

# gallery color prediction via CLIP text classification

def predict_gallery_colors(model, g_paths, I_top, I_bottom):
    top_texts=[f"a person wearing a {c} top" for c in COLORS_11]
    bot_texts=[f"a person wearing {c} pants" for c in COLORS_11]
    Vt=encode_texts(model, top_texts)  # [11,d]
    Vb=encode_texts(model, bot_texts)  # [11,d]
    D={'top':[],'bottom':[]}
    for i in range(len(g_paths)):
        st=(Vt @ I_top[i])  # [11]
        sb=(Vb @ I_bottom[i])  # [11]
        D['top'].append(COLORS_11[int(np.argmax(st))])
        D['bottom'].append(COLORS_11[int(np.argmax(sb))])
    return D

# consistency matrix

def build_consistency_matrix(q_paths, g_paths, q_labels, g_colors):
    Q=len(q_paths); G=len(g_paths)
    C=np.zeros((Q,G), dtype=np.float32)
    for qi,p in enumerate(q_paths):
        lab=q_labels.get(basename_noext(p), {})
        qt=lab.get('top_color','unknown'); qb=lab.get('bottom_color','unknown')
        for gi in range(G):
            denom=0; score=0
            gt=g_colors['top'][gi]; gb=g_colors['bottom'][gi]
            if qt!='unknown' and gt!='unknown':
                denom+=1; score+= int(qt==gt)
            if qb!='unknown' and gb!='unknown':
                denom+=1; score+= int(qb==gb)
            C[qi,gi]= score/denom if denom>0 else 0.0
    return C


def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_lr"; ensure_dir(out_dir)
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    # triwin features
    tri_dir, S_exp_tri, S_attr_tri_mean, S_attr_tri_max = load_triwin_features()
    Z_exp,_m,_s = global_zscore(S_exp_tri)
    Z_attr_mean,_m,_s = global_zscore(S_attr_tri_mean)
    Z_attr_max,_m,_s = global_zscore(S_attr_tri_max)
    # consistency
    model, preprocess=load_clip()
    I_top, I_bottom = encode_gallery_window_embeddings(model, preprocess, g_paths)
    q_labels=load_query_color_labels()
    g_colors=predict_gallery_colors(model, g_paths, I_top, I_bottom)
    write_json(out_dir/'gallery_color_labels.json', {'colors_11':COLORS_11,'top':g_colors['top'],'bottom':g_colors['bottom']})
    C = build_consistency_matrix(q_paths, g_paths, q_labels, g_colors)  # in {0,0.5,1}
    # LR fusion
    Q,G = Z_exp.shape
    # prepare training data
    X=np.stack([Z_exp.flatten(), Z_attr_mean.flatten(), Z_attr_max.flatten(), C.flatten()], axis=1)
    y=np.array([1 if g_ids[j]==q_ids[i] else 0 for i in range(Q) for j in range(G)])
    pos=np.sum(y==1)
    n_folds = int(min(5, max(1,pos)))
    # if pos<2, fallback to single LR fit without CV
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
    if n_folds>=2:
        skf=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        # use CalibratedClassifierCV with sigmoid (Platt)
        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=skf)
        calibrated.fit(X,y)
        probs = calibrated.predict_proba(X)[:,1]
        weights = []
        for cc in getattr(calibrated, 'calibrated_classifiers_', []):
            be = getattr(cc, 'base_estimator', None)
            if be is not None and hasattr(be, 'coef_'):
                weights.append({'coef':be.coef_.tolist(),'intercept':be.intercept_.tolist()})
        calib_info={'method':'platt','folds':n_folds,'pos':int(pos)}
    else:
        clf.fit(X,y)
        probs = clf.predict_proba(X)[:,1]
        weights = {'coef':clf.coef_.tolist(),'intercept':clf.intercept_.tolist()}
        calib_info={'method':'none','folds':1,'pos':int(pos)}
    # reshape fused scores
    S_fuse = probs.reshape(Q,G)
    np.save(out_dir/'S_fuse_lr.npy', S_fuse)
    # metrics
    met = rank1_map_ndcg(S_fuse, q_ids, g_ids)
    write_json(out_dir/'metrics_fuse_lr.json', met)
    write_json(out_dir/'calibration.json', calib_info)
    with open(out_dir/'weights.pkl','wb') as f:
        pickle.dump(weights, f)
    write_csv(out_dir/'summary.csv', [
        ['attrbank_v4_lr', met['Rank-1'], met['mAP'], met['nDCG@10'], 'LR+consistency', 'none']
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    write_json(out_dir/'config.yaml', {
        'features':['z(S_exp_tri)','z(S_attr_tri_mean)','z(S_attr_tri_max)','consistency'],
        'folds':n_folds,
        'class_weight':'balanced',
        'seed':SEED,
        'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},
        'triwin_dir':str(tri_dir)
    })
    # overview append
    overview = OUT_ROOT/f"{ts_tag}_overview.csv"
    write_csv(overview, [[ 'attrbank_v4_lr', met['Rank-1'], met['mAP'], met['nDCG@10'], 'LR+consistency', 'none', '' ]], header=['fusion','Rank-1','mAP','nDCG@10','fusion_detail','rerank','notes'])

if __name__=='__main__':
    main()