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

def encode_images(model, preprocess, paths):
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
    toks=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(toks)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

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
            out[p]=v if v else ['a person']
    return out

# colorcheck outputs

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

# build S_exp using query color labels

def build_S_exp_with_query_colors(q_paths, g_paths, q_color_labels, model, I_base):
    sents_all=[]; idx_map=[]
    for p in q_paths:
        qc=q_color_labels.get(basename_noext(p), {})
        qt=qc.get('top_color','unknown'); qb=qc.get('bottom_color','unknown')
        sents=[]
        if qt!='unknown': sents.append(f"a person wearing a {qt} top")
        if qb!='unknown': sents.append(f"a person wearing {qb} pants")
        if not sents: sents=['a person']
        start=len(sents_all); sents_all.extend(sents); idx_map.append((start,len(sents)))
    V=encode_texts(model, sents_all); S=V@I_base.T
    rows=[np.mean(S[start:start+ln],axis=0) for start,ln in idx_map]
    return np.vstack(rows)

# gallery color prediction and consistency

def predict_gallery_colors(model, I_base):
    top_scores={}
    for c,vars in COLOR_VARIANTS.items():
        sents=[f"a person wearing a {v} top" for v in vars]
        V=encode_texts(model,sents); S=V@I_base.T; top_scores[c]=np.max(S,axis=0)
    bottom_scores={}
    for c,vars in COLOR_VARIANTS.items():
        sents=[f"a person wearing {v} pants" for v in vars]
        V=encode_texts(model,sents); S=V@I_base.T; bottom_scores[c]=np.max(S,axis=0)
    G_top=[]; G_bottom=[]
    for j in range(I_base.shape[0]):
        pt=max([(c,top_scores[c][j]) for c in COLORS_11], key=lambda x:x[1])[0]
        pb=max([(c,bottom_scores[c][j]) for c in COLORS_11], key=lambda x:x[1])[0]
        G_top.append(pt); G_bottom.append(pb)
    return G_top, G_bottom

def build_consistency_matrix_v4(q_paths, g_paths, q_color_labels, model, I_base):
    G_top, G_bottom = predict_gallery_colors(model, I_base)
    C=np.zeros((len(q_paths), len(g_paths)), dtype=np.float32)
    for i,p in enumerate(q_paths):
        qc=q_color_labels.get(basename_noext(p), {})
        qt=qc.get('top_color','unknown'); qb=qc.get('bottom_color','unknown')
        for j in range(len(g_paths)):
            known = (qt!='unknown') + (qb!='unknown')
            if known==0:
                C[i,j]=0.0
            else:
                mt = 1 if (qt!='unknown' and qt==G_top[j]) else 0
                mb = 1 if (qb!='unknown' and qb==G_bottom[j]) else 0
                C[i,j] = (mt+mb)/ known
    return C, G_top, G_bottom

def top1_color_mismatch_ratio(S, q_top, q_bottom, G_top, G_bottom):
    n=S.shape[0]; mism=0; unk=0
    for i in range(n):
        j=int(np.argmax(S[i]))
        mt=(q_top[i] != 'unknown' and G_top[j] != q_top[i])
        mb=(q_bottom[i] != 'unknown' and G_bottom[j] != q_bottom[i])
        if (q_top[i]=='unknown' and q_bottom[i]=='unknown'): unk+=1
        if mt or mb: mism+=1
    return {'top1_mismatch':int(mism),'unknown_queries':int(unk),'total':int(n),'ratio':round(mism/max(1,n),4)}

def append_overview(ts_tag, rows):
    path=OUT_ROOT/f"{ts_tag}_overview.csv"; hdr=['ts','fusion','rerank','gamma','notes','Rank-1','mAP','nDCG@10']
    if not path.exists(): Path(path).write_text(','.join(hdr)+'\n', encoding='utf-8')
    with open(path,'a',encoding='utf-8') as f:
        for r in rows: f.write(','.join(str(x) for x in r)+'\n')

def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_galleryattr"; ensure_dir(out_dir)
    model, preprocess = load_clip()
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    if EMB_IMAGE_PATH.exists():
        I_base=np.load(EMB_IMAGE_PATH)
        if I_base.shape[0]!=len(g_paths): I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    else:
        ensure_dir(EMB_IMAGE_PATH.parent); I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    # query colors
    q_color_labels=load_query_color_labels()
    # S_exp with replaced colors
    S_exp=build_S_exp_with_query_colors(q_paths, g_paths, q_color_labels, model, I_base)
    S_exp_z, _m, _s = global_zscore(S_exp)
    # consistency
    C, G_top, G_bottom = build_consistency_matrix_v4(q_paths, g_paths, q_color_labels, model, I_base)
    # fuse
    gammas=[0.05,0.10,0.15]
    rows=[]
    for gamma in gammas:
        S_fuse=S_exp_z + gamma*C
        met=rank1_map_ndcg(S_fuse, q_ids, g_ids)
        write_json(out_dir/f"metrics_gamma_{gamma}.json", met)
        conf=top1_color_mismatch_ratio(S_fuse, [q_color_labels.get(basename_noext(p),{}).get('top_color','unknown') for p in q_paths], [q_color_labels.get(basename_noext(p),{}).get('bottom_color','unknown') for p in q_paths], G_top, G_bottom)
        write_json(out_dir/f"color_confusion_gamma_{gamma}.json", conf)
        rows.append([ts_tag,'posterior','none',gamma,'v4 colorcheck + small gamma',met['Rank-1'],met['mAP'],met['nDCG@10']])
    # summary
    write_csv(out_dir/'summary.csv', [[f"attrbank_v4_galleryattr_gamma_{g}", *list(json.loads((out_dir/f"metrics_gamma_{g}.json").read_text()).values()), 'posterior','none'] for g in gammas], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    append_overview(ts_tag, rows)
    write_json(out_dir/'config.yaml', {'gamma_list':gammas,'consistency':'normalized_known_sides','filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},'seed':SEED})
    Path(out_dir/'README.md').write_text("v4图库属性后验：使用colorcheck的query颜色与gallery 11色推断，构造一致性矩阵（未知侧不计），S'=z(S_exp)+γ·consistency。", encoding='utf-8')

if __name__=='__main__':
    main()