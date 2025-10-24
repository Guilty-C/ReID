import os, json, time, colorsys
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image
from sklearn.cluster import KMeans

ROOT = Path('d:/PRP SunnyLab/ReID')
OUT_ROOT = ROOT/'outputs/comparison'
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset100.npy'
QUERY_LIST = ROOT/'larger_iso/64/query100.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt'
CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'ViT-L/14'
SEED=42
np.random.seed(SEED)

COLORS_11=['black','white','gray','red','orange','yellow','green','blue','purple','pink','brown']
TOP_TEXTS=[f"a person wearing a {c} top" for c in COLORS_11]
BOT_TEXTS=[f"a person wearing {c} pants" for c in COLORS_11]
TAUS=[0.05,0.10]
GAMMAS=[0.05,0.10,0.15]

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

# windows

def center_square_crop(img):
    w,h=img.size; s=min(w,h); left=(w-s)//2; top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def top_bottom_crops(img):
    sq=center_square_crop(img)
    w,h=sq.size
    return sq.crop((0,0,w,h//2)), sq.crop((0,h//2,w,h))

# encode image windows

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

# HSV primary color via kmeans

def image_primary_color_11(im, sample_max=4096):
    arr=np.array(im)
    h,w,_=arr.shape; flat=arr.reshape(-1,3)
    # sample for speed
    n=min(sample_max, flat.shape[0])
    idx=np.random.choice(flat.shape[0], size=n, replace=False)
    rgb=flat[idx]/255.0
    hsv=np.array([colorsys.rgb_to_hsv(float(r),float(g),float(b)) for r,g,b in rgb])
    km=KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels=km.fit_predict(hsv)
    # pick largest cluster center
    counts=np.bincount(labels)
    center=km.cluster_centers_[int(np.argmax(counts))]
    return map_hsv_to_11(center)

def map_hsv_to_11(hsv):
    h,s,v=float(hsv[0])*360.0, float(hsv[1]), float(hsv[2])
    # achromatic
    if s<0.2:
        if v<0.2: return 'black'
        if v>0.85: return 'white'
        return 'gray'
    # brown heuristic
    if 15<=h<45 and v<0.6: return 'brown'
    # chromatic by hue
    if h<15 or h>=345: return 'red'
    if 15<=h<45: return 'orange'
    if 45<=h<65: return 'yellow'
    if 65<=h<170: return 'green'
    if 170<=h<260: return 'blue'
    if 260<=h<290: return 'purple'
    if 290<=h<345: return 'pink'
    return 'gray'

# CLIP color with margin threshold

def clip_color_from_scores(scores, tau):
    order=np.argsort(-scores)
    top1, top2 = scores[order[0]], scores[order[1]] if len(scores)>1 else -1.0
    if (top1-top2) < tau: return 'unknown'
    return COLORS_11[int(order[0])]

def classify_colors_clip(model, I_top, I_bottom, tau):
    Vt=encode_texts(model, TOP_TEXTS)
    Vb=encode_texts(model, BOT_TEXTS)
    top=[]; bottom=[]
    for i in range(I_top.shape[0]):
        st = Vt @ I_top[i]
        sb = Vb @ I_bottom[i]
        top.append(clip_color_from_scores(st, tau))
        bottom.append(clip_color_from_scores(sb, tau))
    return top, bottom

def encode_texts(model, texts):
    toks=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(toks)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

# combine CLIP + HSV vote

def combined_color_label(top_img, bot_img, clip_top, clip_bot, hsv_top, hsv_bot):
    top_label = clip_top if (clip_top!='unknown' and clip_top==hsv_top) else 'unknown'
    bot_label = clip_bot if (clip_bot!='unknown' and clip_bot==hsv_bot) else 'unknown'
    return top_label, bot_label

# latest triwin S_exp

def latest_triwin_dir():
    dirs=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith('_attrbank_v4_triwin_weighted')]
    return sorted(dirs)[-1] if dirs else None

# explicit caps

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

# fallback compute S_exp_tri if not found

def compute_S_exp_tri(model, preprocess, q_paths, g_paths):
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
    Z_top,_m,_s = global_zscore(S_top)
    Z_full,_m,_s = global_zscore(S_full)
    Z_bot,_m,_s = global_zscore(S_bot)
    S_exp_tri = 0.6*Z_top + 0.3*Z_full + 0.1*Z_bot
    return S_exp_tri

# consistency

def build_consistency(q_paths, g_paths, q_labels, g_labels):
    Q=len(q_paths); G=len(g_paths)
    C=np.zeros((Q,G), dtype=np.float32)
    for i,p in enumerate(q_paths):
        qt=q_labels[basename_noext(p)]['top_color']
        qb=q_labels[basename_noext(p)]['bottom_color']
        for j,gp in enumerate(g_paths):
            gt=g_labels['top'][j]; gb=g_labels['bottom'][j]
            denom=0; score=0
            if qt!='unknown' and gt!='unknown': denom+=1; score+= int(qt==gt)
            if qb!='unknown' and gb!='unknown': denom+=1; score+= int(qb==gb)
            C[i,j]= score/denom if denom>0 else 0.0
    return C

# top1 mismatch under fused ranking

def top1_mismatch_stats(S_fuse, q_paths, g_paths, q_labels, g_labels):
    Q=S_fuse.shape[0]; mism=0; known=0
    for i in range(Q):
        j=int(np.argmax(S_fuse[i]))
        qt=q_labels[basename_noext(q_paths[i])]['top_color']
        qb=q_labels[basename_noext(q_paths[i])]['bottom_color']
        gt=g_labels['top'][j]; gb=g_labels['bottom'][j]
        denom=0; bad=0
        if qt!='unknown' and gt!='unknown': denom+=1; bad+= int(qt!=gt)
        if qb!='unknown' and gb!='unknown': denom+=1; bad+= int(qb!=gb)
        if denom>0:
            known+=1
            if bad>0: mism+=1
    ratio = (mism/known) if known>0 else 0.0
    return {'top1_mismatch':int(mism),'known_queries':int(known),'total':int(Q),'ratio': round(ratio,4)}

# selfsim stats

def img_selfsim_stats(model, preprocess, q_paths):
    I_top, I_full, I_bot = encode_image_windows(model, preprocess, q_paths)
    def cos(a,b): return float(np.dot(a,b))
    sims=[]
    for i,p in enumerate(q_paths):
        sims.append({
            'file': p,
            'sim_top_full': cos(I_top[i], I_full[i]),
            'sim_bottom_full': cos(I_bot[i], I_full[i]),
            'sim_top_bottom': cos(I_top[i], I_bot[i]),
        })
    def agg(key):
        vals=[x[key] for x in sims]
        return {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'p25': float(np.percentile(vals,25)), 'p75': float(np.percentile(vals,75))}
    return {'sim_top_full': agg('sim_top_full'), 'sim_bottom_full': agg('sim_bottom_full'), 'sim_top_bottom': agg('sim_top_bottom'), 'per_query': sims}


def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_colorcons"; ensure_dir(out_dir)
    # filelists
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    write_csv(out_dir/'pos_per_query.csv', [[basename_noext(q_paths[i]), int(np.sum(np.array(g_ids)==q_ids[i]))] for i in range(len(q_paths))], header=['query','pos_in_gallery'])
    # clip + windows
    model, preprocess=load_clip()
    I_top_q, I_full_q, I_bot_q = encode_image_windows(model, preprocess, q_paths)
    I_top_g, I_full_g, I_bot_g = encode_image_windows(model, preprocess, g_paths)
    # HSV for queries/galleries
    def hsv_labels_for_paths(paths):
        t_labels=[]
        b_labels=[]
        for p in paths:
            try:
                img=Image.open(p).convert('RGB')
                sq=center_square_crop(img)
                w,h=sq.size
                top_win=sq.crop((0,0,w,h//2))
                bot_win=sq.crop((0,h//2,w,h))
                t_labels.append(image_primary_color_11(top_win))
                b_labels.append(image_primary_color_11(bot_win))
            except Exception:
                t_labels.append('unknown'); b_labels.append('unknown')
        return t_labels, b_labels
    hsv_top_q, hsv_bot_q = hsv_labels_for_paths(q_paths)
    hsv_top_g, hsv_bot_g = hsv_labels_for_paths(g_paths)
    # run tau grid and pick best by mismatch at gamma=0.10
    best_tau=TAUS[0]; best_ratio=1.0; tau_results={}
    for tau in TAUS:
        clip_top_q, clip_bot_q = classify_colors_clip(model, I_top_q, I_bot_q, tau)
        clip_top_g, clip_bot_g = classify_colors_clip(model, I_top_g, I_bot_g, tau)
        # combine
        q_labels={}
        for idx,p in enumerate(q_paths):
            tq,bq = combined_color_label(None,None, clip_top_q[idx], clip_bot_q[idx], hsv_top_q[idx], hsv_bot_q[idx])
            q_labels[basename_noext(p)]={'top_color':tq,'bottom_color':bq}
        g_labels={'top':[], 'bottom':[]}
        for idx in range(len(g_paths)):
            tg,bg = combined_color_label(None,None, clip_top_g[idx], clip_bot_g[idx], hsv_top_g[idx], hsv_bot_g[idx])
            g_labels['top'].append(tg); g_labels['bottom'].append(bg)
        # consistency
        C=build_consistency(q_paths, g_paths, q_labels, g_labels)
        np.save(out_dir/f'consistency_tau_{str(tau).replace(".","_")}.npy', C)
        # S_exp_tri load or compute
        tri=latest_triwin_dir()
        if tri is not None and (tri/'S_exp_tri.npy').exists():
            S_exp_tri=np.load(tri/'S_exp_tri.npy')
        else:
            S_exp_tri=compute_S_exp_tri(model, preprocess, q_paths, g_paths)
        Z_exp,_m,_s = global_zscore(S_exp_tri)
        # gamma scan
        for gamma in GAMMAS:
            S_fuse = Z_exp + gamma*C
            met = rank1_map_ndcg(S_fuse, q_ids, g_ids)
            write_json(out_dir/f'metrics_gamma_{gamma}_tau_{tau}.json', met)
            conf = top1_mismatch_stats(S_fuse, q_paths, g_paths, q_labels, g_labels)
            write_json(out_dir/f'color_confusion_gamma_{gamma}_tau_{tau}.json', conf)
        # store tau metrics using gamma=0.10 for selection
        S_fuse_10 = Z_exp + 0.10*C
        conf_10 = top1_mismatch_stats(S_fuse_10, q_paths, g_paths, q_labels, g_labels)
        tau_results[str(tau)]=conf_10['ratio']
        if conf_10['ratio']<best_ratio:
            best_ratio=conf_10['ratio']; best_tau=tau
        # also dump tau-specific labels
        write_json(out_dir/f'query_color_labels_tau_{tau}.json', q_labels)
        write_json(out_dir/f'gallery_color_labels_tau_{tau}.json', g_labels)
    # select best tau, re-save canonical labels and consistency
    tau_key=str(best_tau)
    q_labels=json.loads((out_dir/f'query_color_labels_tau_{best_tau}.json').read_text(encoding='utf-8'))
    g_labels=json.loads((out_dir/f'gallery_color_labels_tau_{best_tau}.json').read_text(encoding='utf-8'))
    write_json(out_dir/'query_color_labels.json', q_labels)
    write_json(out_dir/'gallery_color_labels.json', g_labels)
    C_best = np.load(out_dir/f'consistency_tau_{str(best_tau).replace(".","_")}.npy')
    np.save(out_dir/'consistency.npy', C_best)
    # pos_per_query already saved
    # image self-sim stats
    selfsim = img_selfsim_stats(model, preprocess, q_paths)
    write_json(out_dir/'img_selfsim_stats.json', selfsim)
    # summary: list gamma results for best tau
    rows=[]
    for gamma in GAMMAS:
        met=json.loads((out_dir/f'metrics_gamma_{gamma}_tau_{best_tau}.json').read_text(encoding='utf-8'))
        conf=json.loads((out_dir/f'color_confusion_gamma_{gamma}_tau_{best_tau}.json').read_text(encoding='utf-8'))
        rows.append([f'attrbank_v4_1_colorcons_gamma_{gamma}', met['Rank-1'], met['mAP'], met['nDCG@10'], 'colorcons', 'none', conf['ratio']])
    write_csv(out_dir/'summary.csv', rows, header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank','top1_mismatch'])
    # config & run log
    write_json(out_dir/'config.yaml', {'taus':TAUS,'best_tau':best_tau,'gammas':GAMMAS,'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)}})
    Path(out_dir/'run.log').write_text('v4.1 StepA finished.', encoding='utf-8')
    Path(out_dir/'README.md').write_text('v4.1颜色一致性增强：CLIP颜色分类加top1-top2置信阈+HSV主色投票；两者一致才采用，否则unknown。生成labels/consistency并对显式三窗做γ扫描。', encoding='utf-8')
    # overview append
    overview=OUT_ROOT/f"{ts_tag}_overview.csv"
    ov_rows=[[f'v4_1_colorcons', 'none', str(gamma), '-', json.loads((out_dir/f'color_confusion_gamma_{gamma}_tau_{best_tau}.json').read_text())['ratio']] for gamma in GAMMAS]
    write_csv(overview, ov_rows, header=['fusion','rerank','gamma','triwin_w','top1_mismatch'])

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        # write ERRORS.md and still exit
        ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_colorcons"; ensure_dir(out_dir)
        Path(out_dir/'ERRORS.md').write_text(f'Error in StepA: {repr(e)}', encoding='utf-8')
        # overview stub
        overview=OUT_ROOT/f"{ts_tag}_overview.csv"
        write_csv(overview, [[ 'v4_1_colorcons','none','','-','' ]], header=['fusion','rerank','gamma','triwin_w','top1_mismatch'])