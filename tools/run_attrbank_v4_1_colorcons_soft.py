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
THETAS=[0.30,0.35]
DELTAS=[15.0,18.0]
GAMMAS=[0.05,0.08,0.10]
NEIGHBOR_PAIRS={('blue','purple'),('purple','blue'),('orange','brown'),('brown','orange'),('yellow','green'),('green','yellow')}

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

# CLIP

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
            # fallback zeros if read fails
            d=model.visual.output_dim
            I_top.append(np.zeros(d)); I_full.append(np.zeros(d)); I_bottom.append(np.zeros(d))
    return np.stack(I_top), np.stack(I_full), np.stack(I_bottom)

# HSV primary color via kmeans (k=3), with achromatic unify by V

def kmeans_hsv_center(im, sample_max=1024):
    arr=np.array(im)
    h,w,_=arr.shape; flat=arr.reshape(-1,3)
    n=min(sample_max, flat.shape[0])
    idx=np.random.choice(flat.shape[0], size=n, replace=False)
    rgb=flat[idx]/255.0
    hsv=np.array([colorsys.rgb_to_hsv(float(r),float(g),float(b)) for r,g,b in rgb])
    km=KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels=km.fit_predict(hsv)
    counts=np.bincount(labels)
    center=km.cluster_centers_[int(np.argmax(counts))]
    return center  # (h in [0,1], s, v)


def hue_center_for_color(color):
    # approximate canonical hue centers (degrees)
    mapping={'red':0.0,'orange':30.0,'yellow':55.0,'green':120.0,'blue':210.0,'purple':275.0,'pink':315.0}
    return mapping.get(color, None)


def map_hsv_to_11(center):
    h_deg=float(center[0])*360.0; s=float(center[1]); v=float(center[2])
    # achromatic unify by V
    if v>=0.80: return 'white'
    if v<=0.20: return 'black'
    if s<0.20: return 'gray'
    # brown heuristic
    if 15<=h_deg<45 and v<0.6: return 'brown'
    # chromatic by hue
    if h_deg<15 or h_deg>=345: return 'red'
    if 15<=h_deg<45: return 'orange'
    if 45<=h_deg<65: return 'yellow'
    if 65<=h_deg<170: return 'green'
    if 170<=h_deg<260: return 'blue'
    if 260<=h_deg<290: return 'purple'
    if 290<=h_deg<345: return 'pink'
    return 'gray'

# CLIP color scores

def encode_texts(model, texts):
    toks=clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        v=model.encode_text(toks).float()
    v=v/ v.norm(dim=-1, keepdim=True)
    return v  # keep on DEVICE


def clip_color_scores(model, I_vec, texts):
    V=encode_texts(model, texts)
    return V @ I_vec  # shape [len(texts)]


def top_two(scores):
    # support numpy or torch tensor
    if isinstance(scores, np.ndarray):
        arr = scores
    elif torch.is_tensor(scores):
        arr = scores.detach().cpu().numpy()
    else:
        arr = np.array(scores)
    order=np.argsort(-arr)
    a= arr[order[0]]
    b= arr[order[1]] if len(arr)>1 else -1.0
    idx=int(order[0])
    return a,b, COLORS_11[idx]

# soft combine rule

def soft_label(clip_scores, hsv_label, hsv_center, theta, tau, delta):
    top1, top2, clip_lab = top_two(clip_scores)
    # high confidence CLIP override
    if (top1>=theta) and ((top1-top2)>=0.03):
        return clip_lab
    # else require agreement or neighbor (with hue proximity)
    if (top1-top2) < tau:
        # low margin -> trust HSV only if chromatic
        return 'unknown' if hsv_label=='gray' else hsv_label
    if clip_lab==hsv_label:
        return clip_lab
    # neighbor check
    if (clip_lab, hsv_label) in NEIGHBOR_PAIRS:
        hc = hue_center_for_color(clip_lab)
        if hc is not None:
            h = float(hsv_center[0])*360.0
            # circular hue distance
            dh = min(abs(h-hc), 360.0-abs(h-hc))
            if dh <= delta:
                return clip_lab
    return 'unknown'

# compute labels for paths

# (kept for reference but not used in the optimized path)
def compute_labels_soft(model, preprocess, paths, theta, tau, delta, V_TOP, V_BOT):
    top_labels=[]; bot_labels=[]; hsv_centers_top=[]; hsv_centers_bot=[]
    for p in paths:
        try:
            img=Image.open(p).convert('RGB')
        except Exception:
            top_labels.append('unknown'); bot_labels.append('unknown')
            hsv_centers_top.append((0.0,0.0,0.0)); hsv_centers_bot.append((0.0,0.0,0.0))
            continue
        top_img, bot_img = top_bottom_crops(img)
        # HSV centers
        hc_top = kmeans_hsv_center(top_img, sample_max=512)
        hc_bot = kmeans_hsv_center(bot_img, sample_max=512)
        hsv_top = map_hsv_to_11(hc_top)
        hsv_bot = map_hsv_to_11(hc_bot)
        # CLIP window embeddings
        x_top=preprocess(top_img).unsqueeze(0).to(DEVICE)
        x_bot=preprocess(bot_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vT = model.encode_image(x_top).float()
            vB = model.encode_image(x_bot).float()
        vT = vT/ vT.norm(dim=-1, keepdim=True)
        vB = vB/ vB.norm(dim=-1, keepdim=True)
        # use precomputed text embeddings
        s_top = V_TOP @ vT.squeeze(0).cpu().numpy()
        s_bot = V_BOT @ vB.squeeze(0).cpu().numpy()
        # soft combine
        lab_top = soft_label(s_top, hsv_top, hc_top, theta, tau, delta)
        lab_bot = soft_label(s_bot, hsv_bot, hc_bot, theta, tau, delta)
        top_labels.append(lab_top); bot_labels.append(lab_bot)
        hsv_centers_top.append(hc_top); hsv_centers_bot.append(hc_bot)
    return top_labels, bot_labels, hsv_centers_top, hsv_centers_bot

# Optimized: precompute features once per path

def precompute_color_features(model, preprocess, paths, V_TOP, V_BOT, sample_max=512):
    scores_top=[]; scores_bot=[]; hc_top_list=[]; hc_bot_list=[]; hsv_top_list=[]; hsv_bot_list=[]
    for p in paths:
        try:
            img=Image.open(p).convert('RGB')
        except Exception:
            d=model.visual.output_dim
            vec_zero=torch.zeros(d, dtype=torch.float32, device=DEVICE)
            scores_top.append(torch.matmul(V_TOP, vec_zero).detach())
            scores_bot.append(torch.matmul(V_BOT, vec_zero).detach())
            hc_top_list.append((0.0,0.0,0.0)); hc_bot_list.append((0.0,0.0,0.0))
            hsv_top_list.append('gray'); hsv_bot_list.append('gray')
            continue
        top_img, bot_img = top_bottom_crops(img)
        # HSV
        hc_top = kmeans_hsv_center(top_img, sample_max=sample_max)
        hc_bot = kmeans_hsv_center(bot_img, sample_max=sample_max)
        hsv_top = map_hsv_to_11(hc_top)
        hsv_bot = map_hsv_to_11(hc_bot)
        # window embeddings
        x_top=preprocess(top_img).unsqueeze(0).to(DEVICE)
        x_bot=preprocess(bot_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vT = model.encode_image(x_top).float()
            vB = model.encode_image(x_bot).float()
        vT = vT/ vT.norm(dim=-1, keepdim=True)
        vB = vB/ vB.norm(dim=-1, keepdim=True)
        vecT=vT.squeeze(0)
        vecB=vB.squeeze(0)
        # scores on GPU
        scores_top.append(torch.matmul(V_TOP, vecT).detach())
        scores_bot.append(torch.matmul(V_BOT, vecB).detach())
        # keep hsv info
        hc_top_list.append(hc_top); hc_bot_list.append(hc_bot)
        hsv_top_list.append(hsv_top); hsv_bot_list.append(hsv_bot)
    return {
        'scores_top': torch.stack(scores_top),
        'scores_bot': torch.stack(scores_bot),
        'hc_top': hc_top_list,
        'hc_bot': hc_bot_list,
        'hsv_top': hsv_top_list,
        'hsv_bot': hsv_bot_list,
    }


def labels_from_precomputed(precomp, theta, tau, delta):
    top_labels=[]; bot_labels=[]
    for i in range(len(precomp['hsv_top'])):
        s_top = precomp['scores_top'][i]
        s_bot = precomp['scores_bot'][i]
        hsv_t = precomp['hsv_top'][i]; hsv_b = precomp['hsv_bot'][i]
        hc_t = precomp['hc_top'][i]; hc_b = precomp['hc_bot'][i]
        lab_top = soft_label(s_top, hsv_t, hc_t, theta, tau, delta)
        lab_bot = soft_label(s_bot, hsv_b, hc_b, theta, tau, delta)
        top_labels.append(lab_top); bot_labels.append(lab_bot)
    return top_labels, bot_labels
# consistency matrix (0,0.5,1)

def build_consistency(q_paths, g_paths, q_labels, g_labels):
    Q=len(q_paths); G=len(g_paths)
    C=np.zeros((Q,G), dtype=np.float32)
    for i,p in enumerate(q_paths):
        qt=q_labels[basename_noext(p)]['top_color']
        qb=q_labels[basename_noext(p)]['bottom_color']
        for j,_ in enumerate(g_paths):
            gt=g_labels['top'][j]; gb=g_labels['bottom'][j]
            denom=0; score=0
            if qt!='unknown' and gt!='unknown': denom+=1; score+= int(qt==gt)
            if qb!='unknown' and gb!='unknown': denom+=1; score+= int(qb==gb)
            C[i,j]= score/denom if denom>0 else 0.0
    return C

# triwin explicit

def latest_triwin_dir():
    dirs=[d for d in OUT_ROOT.iterdir() if d.is_dir() and d.name.endswith('_attrbank_v4_triwin_weighted')]
    return sorted(dirs)[-1] if dirs else None


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

# top1 mismatch over queries with both-known top&bottom

def top1_mismatch_stats(S_fuse, q_paths, g_paths, q_labels, g_labels):
    Q=S_fuse.shape[0]; mism=0; known_both=0
    for i in range(Q):
        j=int(np.argmax(S_fuse[i]))
        qt=q_labels[basename_noext(q_paths[i])]['top_color']
        qb=q_labels[basename_noext(q_paths[i])]['bottom_color']
        gt=g_labels['top'][j]; gb=g_labels['bottom'][j]
        if qt!='unknown' and qb!='unknown' and gt!='unknown' and gb!='unknown':
            known_both+=1
            if qt!=gt or qb!=gb: mism+=1
    ratio = (mism/known_both) if known_both>0 else 0.0
    return {'top1_mismatch':int(mism),'known_both':int(known_both),'total':int(Q),'ratio': round(ratio,4)}

# known coverage (at least one known per query)

def known_coverage_rate(q_paths, q_labels):
    known_any=0
    for p in q_paths:
        lab=q_labels[basename_noext(p)]
        if lab['top_color']!='unknown' or lab['bottom_color']!='unknown':
            known_any+=1
    return known_any/len(q_paths)


def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_colorcons_soft"; ensure_dir(out_dir)
    errors=[]
    # filelists
    q_paths=read_lines(QUERY_LIST); g_paths=read_lines(GALLERY_LIST)
    q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    write_csv(out_dir/'pos_per_query.csv', [[basename_noext(q_paths[i]), int(np.sum(np.array(g_ids)==q_ids[i]))] for i in range(len(q_paths))], header=['query','pos_in_gallery'])
    if not EMB_IMAGE_PATH.exists(): errors.append(f"Missing embed file: {EMB_IMAGE_PATH}")
    # clip
    model, preprocess=load_clip()
    # precompute text embeddings for top/bottom color prompts
    V_TOP = encode_texts(model, TOP_TEXTS)
    V_BOT = encode_texts(model, BOT_TEXTS)
    # triwin explicit
    tri=latest_triwin_dir()
    if tri is not None and (tri/'S_exp_tri.npy').exists():
        S_exp_tri=np.load(tri/'S_exp_tri.npy')
    else:
        # minimal fallback: zero matrix
        S_exp_tri=np.zeros((len(q_paths), len(g_paths)), dtype=np.float32); errors.append('Missing S_exp_tri.npy; using zeros')
    Z_exp,_m,_s = global_zscore(S_exp_tri)
    # soft label combos scan: select best by mismatch ratio (gamma=0.10), tie-break by known_rate desc
    best=(None,None,None); best_ratio=1.0; best_known=0.0
    combo_results={}
    # precompute window features once (GPU)
    q_pre = precompute_color_features(model, preprocess, q_paths, V_TOP, V_BOT, sample_max=512)
    g_pre = precompute_color_features(model, preprocess, g_paths, V_TOP, V_BOT, sample_max=512)
    for tau in TAUS:
        for theta in THETAS:
            for delta in DELTAS:
                # labels from precomputed
                q_top, q_bot = labels_from_precomputed(q_pre, theta, tau, delta)
                g_top, g_bot = labels_from_precomputed(g_pre, theta, tau, delta)
                q_labels={basename_noext(q_paths[i]): {'top_color':q_top[i],'bottom_color':q_bot[i]} for i in range(len(q_paths))}
                g_labels={'top': g_top, 'bottom': g_bot}
                # consistency
                C=build_consistency(q_paths, g_paths, q_labels, g_labels)
                np.save(out_dir/f'consistency_tau_{str(tau).replace(".","_")}_theta_{str(theta).replace(".","_")}_delta_{int(delta)}.npy', C)
                # gamma scan with per-query gamma=0 if both unknown
                known_rate = known_coverage_rate(q_paths, q_labels)
                S_rows=[]
                for gamma in GAMMAS:
                    gamma_vec = np.array([0.0 if (q_labels[basename_noext(p)]['top_color']=='unknown' and q_labels[basename_noext(p)]['bottom_color']=='unknown') else gamma for p in q_paths], dtype=np.float32)
                    S_fuse = Z_exp + (gamma_vec[:,None]*C)
                    met = rank1_map_ndcg(S_fuse, q_ids, g_ids)
                    write_json(out_dir/f'metrics_gamma_{gamma}_tau_{tau}_theta_{theta}_delta_{int(delta)}.json', met)
                    conf = top1_mismatch_stats(S_fuse, q_paths, g_paths, q_labels, g_labels)
                    conf['known_rate']= round(known_rate,4)
                    write_json(out_dir/f'color_confusion_gamma_{gamma}_tau_{tau}_theta_{theta}_delta_{int(delta)}.json', conf)
                    S_rows.append((gamma, met, conf))
                # choose best combo at gamma=0.10
                for gamma, met, conf in S_rows:
                    if abs(gamma-0.10)<1e-8:
                        ratio=conf['ratio']; kr=conf['known_rate']
                        combo_results[(tau,theta,delta)]={'known_rate':kr,'mismatch':ratio}
                        if (ratio<best_ratio) or (abs(ratio-best_ratio)<1e-8 and kr>best_known):
                            best_ratio=ratio; best_known=kr; best=(tau,theta,delta)
                # dump labels per combo for traceability
                write_json(out_dir/f'query_color_labels_tau_{tau}_theta_{theta}_delta_{int(delta)}.json', q_labels)
                write_json(out_dir/f'gallery_color_labels_tau_{tau}_theta_{theta}_delta_{int(delta)}.json', g_labels)
    # select best combo and save canonical
    tau_best, theta_best, delta_best = best
    q_labels=json.loads((out_dir/f'query_color_labels_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text(encoding='utf-8'))
    g_labels=json.loads((out_dir/f'gallery_color_labels_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text(encoding='utf-8'))
    write_json(out_dir/'query_color_labels.json', q_labels)
    write_json(out_dir/'gallery_color_labels.json', g_labels)
    C_best = np.load(out_dir/f'consistency_tau_{str(tau_best).replace(".","_")}_theta_{str(theta_best).replace(".","_")}_delta_{int(delta_best)}.npy')
    np.save(out_dir/'consistency.npy', C_best)
    # image self-sim stats (optional diagnostic)
    def cos(a,b): return float(np.dot(a,b))
    I_top_q, I_full_q, I_bot_q = encode_image_windows(model, preprocess, q_paths)
    sims=[]
    for i,p in enumerate(q_paths): sims.append({'file':p,'sim_top_full':cos(I_top_q[i],I_full_q[i]),'sim_bottom_full':cos(I_bot_q[i],I_full_q[i]),'sim_top_bottom':cos(I_top_q[i],I_bot_q[i])})
    def agg(key):
        vals=[x[key] for x in sims]
        return {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'p25': float(np.percentile(vals,25)), 'p75': float(np.percentile(vals,75))}
    write_json(out_dir/'img_selfsim_stats.json', {'sim_top_full': agg('sim_top_full'), 'sim_bottom_full': agg('sim_bottom_full'), 'sim_top_bottom': agg('sim_top_bottom'), 'per_query': sims})
    # summary rows for best combo
    rows=[]
    for gamma in GAMMAS:
        met=json.loads((out_dir/f'metrics_gamma_{gamma}_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text(encoding='utf-8'))
        conf=json.loads((out_dir/f'color_confusion_gamma_{gamma}_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text(encoding='utf-8'))
        rows.append([f'attrbank_v4_1_colorcons_soft_gamma_{gamma}', met['Rank-1'], met['mAP'], met['nDCG@10'], 'colorcons_soft', 'none', conf['known_rate'], conf['ratio']])
    write_csv(out_dir/'summary.csv', rows, header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank','known_rate','top1_mismatch'])
    # config & run log
    write_json(out_dir/'config.yaml', {'taus':TAUS,'thetas':THETAS,'deltas':DELTAS,'best':{'tau':tau_best,'theta':theta_best,'delta':delta_best},'gammas':GAMMAS,'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},'embeds':str(EMB_IMAGE_PATH)})
    Path(out_dir/'run.log').write_text('v4.1-soft StepA finished.', encoding='utf-8')
    Path(out_dir/'README.md').write_text('v4.1-soft 颜色一致性：高置信CLIP直采；否则CLIP+HSV一致或邻近(Δh≤δ)为已知；非彩色统一V阈；unknown不计分；γ按query是否两项均unknown置0。', encoding='utf-8')
    # overview append
    overview=OUT_ROOT/f"{ts_tag}_overview.csv"
    ov_rows=[[f'v4_1_colorcons_soft', 'none', str(gamma), '-', json.loads((out_dir/f'color_confusion_gamma_{gamma}_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text())['known_rate'], json.loads((out_dir/f'color_confusion_gamma_{gamma}_tau_{tau_best}_theta_{theta_best}_delta_{int(delta_best)}.json').read_text())['ratio']] for gamma in GAMMAS]
    write_csv(overview, ov_rows, header=['fusion','rerank','gamma','triwin_w','known_rate','top1_mismatch'])
    if errors:
        Path(out_dir/'ERRORS.md').write_text("\n".join(errors), encoding='utf-8')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_1_colorcons_soft"; ensure_dir(out_dir)
        Path(out_dir/'ERRORS.md').write_text(f'Error in StepA-soft: {repr(e)}', encoding='utf-8')
        overview=OUT_ROOT/f"{ts_tag}_overview.csv"
        write_csv(overview, [[ 'v4_1_colorcons_soft','none','','-','','' ]], header=['fusion','rerank','gamma','triwin_w','known_rate','top1_mismatch'])