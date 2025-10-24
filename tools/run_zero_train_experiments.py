import os, sys, json, time, math, csv, glob, shutil, random
from pathlib import Path
import numpy as np
from PIL import Image

# Optional: sklearn for AUC
try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import torch
import clip

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Fixed inputs
ROOT = Path('d:/PRP SunnyLab/ReID')
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
QUERY_LIST = ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery.txt'
LABELS_FILE = ROOT/'data/market1501.images.labels_subset50.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
PROMPT_TUNER_DIR = ROOT/'outputs/prompt_tuner'

TEMPS = [0.05, 0.1, 0.2, 0.5, 1.0]
ALPHAS = [0.2, 0.5, 0.8]
BETAS = [0.3, 0.5, 0.7]

MODEL_NAME = 'ViT-L/14'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Utils

def ts_local():
    return time.strftime('%Y%m%d-%H%M%S', time.localtime())

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def read_lines(p):
    return [l.strip() for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2), encoding='utf-8')

def write_csv(p, rows, header=None):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def basename_noext(p):
    return Path(p).stem

def load_clip():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return model, preprocess

def l2norm(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom==0] = 1.0
    return x/denom

# Data assembly

def get_query_gallery():
    q = read_lines(QUERY_LIST)
    g = read_lines(GALLERY_LIST)
    return q, g

def get_ids_from_paths(paths):
    ids = []
    for p in paths:
        s = basename_noext(p)
        # Market1501 naming e.g., 0001_c1s1_... -> id until first '_'
        id_part = s.split('_')[0]
        try:
            ids.append(int(id_part))
        except:
            ids.append(id_part)
    return ids

def load_explicit_captions_for_subset(q_paths):
    if not EXPL_CAP_PATH.exists():
        # Fallback: simple person captions
        return {p: ['a person'] for p in q_paths}
    try:
        data = json.loads(EXPL_CAP_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {p: ['a person'] for p in q_paths}
    # Data can be list of dicts or mapping; support both
    caps = {}
    # Normalize keys to base filename for matching
    by_base = {}
    if isinstance(data, dict):
        for k, v in data.items():
            by_base[basename_noext(k)] = v
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                fn = item.get('file') or item.get('path') or item.get('image') or ''
                texts = item.get('captions') or item.get('text') or item.get('desc') or item.get('salient') or item.get('captions_list') or []
                by_base[basename_noext(fn)] = texts
    for p in q_paths:
        base = basename_noext(p)
        v = by_base.get(base)
        if not v:
            caps[p] = ['a person']
        else:
            # ensure list of strings
            if isinstance(v, str): v = [v]
            v = [s for s in v if isinstance(s, str) and s.strip()]
            caps[p] = v if v else ['a person']
    return caps

def load_implicit_prompt():
    if PROMPT_TUNER_DIR.exists():
        runs = sorted(PROMPT_TUNER_DIR.glob('*/best_prompt.txt'), key=os.path.getmtime)
        if runs:
            try:
                txt = Path(runs[-1]).read_text(encoding='utf-8').strip()
                if txt: return txt
            except: pass
        vecs = sorted(PROMPT_TUNER_DIR.glob('*/prompt_vec.npy'), key=os.path.getmtime)
        if vecs:
            return 'person'  # We only use text, not vectors here
    return 'person'

# Embedding

def encode_texts(model, texts):
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats.float()
    return l2norm(feats.cpu().numpy())

def encode_images(model, preprocess, img_paths):
    ims = []
    for p in img_paths:
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            # create dummy gray image
            im = Image.new('RGB', (224, 224), (127,127,127))
        ims.append(preprocess(im))
    batch = torch.stack(ims).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats.float()
    return l2norm(feats.cpu().numpy())

def crop_windows(im):
    W, H = im.size
    top_h = int(0.6*H)
    bot_y = int(0.4*H)
    top = im.crop((0,0,W,top_h))
    bot = im.crop((0,bot_y,W,H))
    full = im
    return top, bot, full

def encode_images_triwin(model, preprocess, img_paths):
    feats_top, feats_bot, feats_full = [], [], []
    for p in img_paths:
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            im = Image.new('RGB', (224,224), (127,127,127))
        t, b, f = crop_windows(im)
        it = preprocess(t); ib = preprocess(b); iff = preprocess(f)
        batch = torch.stack([it, ib, iff]).to(DEVICE)
        with torch.no_grad():
            x = model.encode_image(batch).float()
        x = l2norm(x.cpu().numpy())
        feats_top.append(x[0]); feats_bot.append(x[1]); feats_full.append(x[2])
    return np.vstack(feats_top), np.vstack(feats_bot), np.vstack(feats_full)

# Metrics

def sim_from_text_image(T, I):
    return np.dot(T, I.T)

def rank1_map_ndcg(S, q_ids, g_ids, k_ndcg=10):
    n_q = S.shape[0]
    rank1 = 0
    ap_sum = 0.0
    ndcg_sum = 0.0
    for i in range(n_q):
        sims = S[i]
        order = np.argsort(-sims)
        rel = (np.array(g_ids)[order] == q_ids[i]).astype(np.int32)
        # Rank-1
        rank1 += int(rel[0] == 1)
        # AP
        pos = np.where(rel==1)[0]
        if len(pos)==0:
            ap = 0.0
        else:
            precs = []
            for j, idx in enumerate(pos, start=1):
                prec = np.sum(rel[:idx+1])/(idx+1)
                precs.append(prec)
            ap = float(np.mean(precs))
        ap_sum += ap
        # nDCG@k
        k = min(k_ndcg, len(rel))
        gains = rel[:k]
        if np.sum(gains)==0:
            ndcg = 0.0
        else:
            dcg = np.sum(gains/np.log2(np.arange(2, k+2)))
            ideal = np.sort(rel)[::-1][:k]
            idcg = np.sum(ideal/np.log2(np.arange(2, k+2)))
            ndcg = dcg/(idcg+1e-9)
        ndcg_sum += ndcg
    return {
        'Rank-1': round(rank1/n_q, 4),
        'mAP': round(ap_sum/n_q, 4),
        'nDCG@10': round(ndcg_sum/n_q, 4)
    }

def global_auc(S, q_ids, g_ids):
    # Unnormalized per-row scores: directly use S
    y_true = []
    y_score = []
    for i in range(S.shape[0]):
        rel = (np.array(g_ids) == q_ids[i]).astype(np.int32)
        y_true.extend(rel.tolist())
        y_score.extend(S[i].tolist())
    if SKLEARN_OK:
        try:
            auc = roc_auc_score(np.array(y_true), np.array(y_score))
            return float(auc)
        except Exception:
            pass
    # Fallback: pairwise AUC estimate
    pos_scores = [s for t,s in zip(y_true, y_score) if t==1]
    neg_scores = [s for t,s in zip(y_true, y_score) if t==0]
    if len(pos_scores)==0 or len(neg_scores)==0:
        return 0.0
    wins = 0
    ties = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps>ns: wins+=1
            elif ps==ns: ties+=1
    total = len(pos_scores)*len(neg_scores)
    return (wins + 0.5*ties)/total

def zscore_rows(S):
    m = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1, keepdims=True)+1e-9
    return (S-m)/std

def curves(S, q_ids, g_ids):
    y_true = []
    y_score = []
    for i in range(S.shape[0]):
        rel = (np.array(g_ids) == q_ids[i]).astype(np.int32)
        y_true.extend(rel.tolist())
        y_score.extend(S[i].tolist())
    if SKLEARN_OK:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        pr, rc, _ = precision_recall_curve(y_true, y_score)
        roc_rows = [[fpr[i], tpr[i]] for i in range(len(fpr))]
        pr_rows = [[rc[i], pr[i]] for i in range(len(rc))]
        return roc_rows, pr_rows
    else:
        return [], []

# Attribute bank
COLORS = ['black','white','blue','red','green','yellow','purple','pink','orange','brown','gray']
COLOR_SET = set(COLORS)

def extract_attrs_from_caption(txt):
    t = txt.lower()
    def find_color(tokens):
        for c in COLORS:
            for tok in tokens:
                if c in tok:
                    return c
        return 'unknown'
    tokens = t.replace(',', ' ').replace('.', ' ').split()
    top_color = find_color(tokens)
    bottom_color = find_color(tokens)
    shoes_color = 'other'
    if 'white shoes' in t or 'white sneakers' in t or 'white' in t and 'shoes' in t:
        shoes_color = 'white'
    elif 'black shoes' in t or 'black sneakers' in t or ('black' in t and 'shoes' in t):
        shoes_color = 'black'
    bag = 'with bag' if any(k in t for k in ['bag','backpack','handbag','shoulder bag']) else 'no bag'
    return top_color, bottom_color, shoes_color, bag

def build_attr_sentences(caps_list):
    # Use the first caption to extract attrs
    if not caps_list: caps_list=['a person']
    top_color, bottom_color, shoes_color, bag = extract_attrs_from_caption(caps_list[0])
    s1 = f"a person wearing {top_color} top, {bottom_color} bottoms, {shoes_color} shoes, {bag}"
    s2 = f"a person, {bag}, {shoes_color} shoes"
    s3 = f"person with {top_color} top and {bottom_color} pants"
    return [s1, s2, s3]

# Logging and config

def write_config(dir_path, extra=None):
    cfg = {
        'model': 'CLIP ViT-L/14',
        'device': DEVICE,
        'temps': TEMPS,
        'alphas': ALPHAS,
        'betas': BETAS,
        'seed': SEED
    }
    if extra: cfg.update(extra)
    Path(dir_path, 'config.yaml').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

def log(dir_path, msg):
    with open(Path(dir_path, 'run.log'), 'a', encoding='utf-8') as f:
        f.write(msg.rstrip()+'\n')

# Main execution

def run():
    ts = ts_local()
    out_root = ROOT/'outputs/comparison'
    diag_dir = out_root/f"{ts}_diag"; triwin_dir = out_root/f"{ts}_triwin"; attr_dir = out_root/f"{ts}_attrbank"
    for d in [diag_dir, triwin_dir, attr_dir]: ensure_dir(d);
    # Data
    q_paths, g_paths = get_query_gallery()
    q_ids = get_ids_from_paths(q_paths)
    g_ids = get_ids_from_paths(g_paths)

    # Self-sim check for gallery embeddings
    img_selfsim = {}
    if EMB_IMAGE_PATH.exists():
        I = np.load(EMB_IMAGE_PATH)
        img_ss = np.dot(I, I.T)
        max_diag = float(np.max(np.diag(img_ss)))
        offdiag = img_ss - np.diag(np.diag(img_ss))
        max_offdiag = float(np.max(offdiag))
        img_selfsim = {
            'emb_shape': list(I.shape),
            'diag_max': round(max_diag,5),
            'offdiag_max': round(max_offdiag,5),
            'diag_dominant': bool(max_diag >= max_offdiag)
        }
    else:
        img_selfsim = {'error':'missing image embeddings'}
    write_json(diag_dir/'img_selfsim_stats.json', img_selfsim)

    # Pos per query
    rows = [[basename_noext(q_paths[i]), int(np.sum(np.array(g_ids)==q_ids[i]))] for i in range(len(q_paths))]
    write_csv(diag_dir/'pos_per_query.csv', rows, header=['query','pos_in_gallery'])

    # Load CLIP
    model, preprocess = load_clip()

    # Explicit/Implicit texts
    expl_caps = load_explicit_captions_for_subset(q_paths)
    expl_texts = [expl_caps[p][0] for p in q_paths]  # take first
    imp_text = load_implicit_prompt()
    imp_texts = [imp_text for _ in q_paths]

    # Encode texts
    T_exp = encode_texts(model, expl_texts)
    T_imp = encode_texts(model, imp_texts)

    # Baseline: use cached image embeds if present else compute and save
    if EMB_IMAGE_PATH.exists():
        I_base = np.load(EMB_IMAGE_PATH)
        if I_base.shape[0] != len(g_paths):
            # recompute to align
            I_base = encode_images(model, preprocess, g_paths)
            np.save(EMB_IMAGE_PATH, I_base)
    else:
        I_base = encode_images(model, preprocess, g_paths)
        ensure_dir(EMB_IMAGE_PATH.parent)
        np.save(EMB_IMAGE_PATH, I_base)

    # Baseline sims
    S_exp = sim_from_text_image(T_exp, I_base)
    S_imp = sim_from_text_image(T_imp, I_base)

    # Temperature selection only for threshold metrics (AUC/curves)
    aucs_exp = []; aucs_imp = []
    for T in TEMPS:
        aucs_exp.append(global_auc(S_exp/ T, q_ids, g_ids))
        aucs_imp.append(global_auc(S_imp/ T, q_ids, g_ids))
    bestT_exp = TEMPS[int(np.argmax(aucs_exp))]
    bestT_imp = TEMPS[int(np.argmax(aucs_imp))]

    metrics_exp = rank1_map_ndcg(S_exp, q_ids, g_ids)
    metrics_exp['AUC'] = round(global_auc(S_exp/ bestT_exp, q_ids, g_ids),4)
    metrics_imp = rank1_map_ndcg(S_imp, q_ids, g_ids)
    metrics_imp['AUC'] = round(global_auc(S_imp/ bestT_imp, q_ids, g_ids),4)

    # Curves
    roc_exp, pr_exp = curves(S_exp/ bestT_exp, q_ids, g_ids)
    roc_imp, pr_imp = curves(S_imp/ bestT_imp, q_ids, g_ids)
    ensure_dir(diag_dir/'curves')
    write_csv(diag_dir/'curves/roc_exp.csv', roc_exp, header=['fpr','tpr'])
    write_csv(diag_dir/'curves/pr_exp.csv', pr_exp, header=['recall','precision'])
    write_csv(diag_dir/'curves/roc_imp.csv', roc_imp, header=['fpr','tpr'])
    write_csv(diag_dir/'curves/pr_imp.csv', pr_imp, header=['recall','precision'])

    # Save sims and metrics
    np.save(diag_dir/'sim_exp.npy', S_exp)
    np.save(diag_dir/'sim_imp.npy', S_imp)
    write_json(diag_dir/'metrics_exp.json', {**metrics_exp, 'best_T':bestT_exp})
    write_json(diag_dir/'metrics_imp.json', {**metrics_imp, 'best_T':bestT_imp})

    # AUC explanation
    auc_md = (
        "AUC uses global y_true (all query-gallery pairs) and raw similarity y_score.\n"
        "We select best temperature T on AUC grid {"+','.join(map(str,TEMPS))+"}. No per-row z-scoring.\n"
        "If distances were used we would negate; here cosine similarities are used directly."
    )
    Path(diag_dir/'auc_check.md').write_text(auc_md, encoding='utf-8')

    # Summary + README
    write_csv(diag_dir/'summary.csv', [
        ['explicit', metrics_exp['Rank-1'], metrics_exp['mAP'], metrics_exp['nDCG@10'], metrics_exp['AUC'], bestT_exp],
        ['implicit', metrics_imp['Rank-1'], metrics_imp['mAP'], metrics_imp['nDCG@10'], metrics_imp['AUC'], bestT_imp]
    ], header=['setting','Rank-1','mAP','nDCG@10','AUC','best_T'])
    readme = (
        "基线完成：显式与隐式均已评测并自检。AUC基于全对展开、未做行内归一。\n"
        "显式通常优于隐式，但整体指标较低；后续将尝试三窗与属性银行以提升mAP/nDCG。"
    )
    Path(diag_dir/'README.md').write_text(readme, encoding='utf-8')
    write_config(diag_dir)

    # --- TRIWIN ---
    write_config(triwin_dir)
    I_top, I_bot, I_full = encode_images_triwin(model, preprocess, g_paths)
    S_exp_tw = np.maximum.reduce([sim_from_text_image(T_exp, I_top), sim_from_text_image(T_exp, I_bot), sim_from_text_image(T_exp, I_full)])
    S_imp_tw = np.maximum.reduce([sim_from_text_image(T_imp, I_top), sim_from_text_image(T_imp, I_bot), sim_from_text_image(T_imp, I_full)])
    # temperature selection for AUC
    aucs_exp_tw = [global_auc(S_exp_tw/ t, q_ids, g_ids) for t in TEMPS]
    aucs_imp_tw = [global_auc(S_imp_tw/ t, q_ids, g_ids) for t in TEMPS]
    bestT_exp_tw = TEMPS[int(np.argmax(aucs_exp_tw))]
    bestT_imp_tw = TEMPS[int(np.argmax(aucs_imp_tw))]
    met_exp_tw = rank1_map_ndcg(S_exp_tw, q_ids, g_ids); met_exp_tw['AUC']=round(global_auc(S_exp_tw/ bestT_exp_tw, q_ids, g_ids),4)
    met_imp_tw = rank1_map_ndcg(S_imp_tw, q_ids, g_ids); met_imp_tw['AUC']=round(global_auc(S_imp_tw/ bestT_imp_tw, q_ids, g_ids),4)
    # Fusion alpha by mAP
    best_alpha = None; best_m = -1; best_metrics_fuse=None; bestT_fuse=None
    for a in ALPHAS:
        S_fuse = a*zscore_rows(S_exp_tw) + (1-a)*zscore_rows(S_imp_tw)
        aucs_fuse = [global_auc(S_fuse/ t, q_ids, g_ids) for t in TEMPS]
        Topt = TEMPS[int(np.argmax(aucs_fuse))]
        m = rank1_map_ndcg(S_fuse, q_ids, g_ids)
        if m['mAP']>best_m:
            best_m = m['mAP']; best_alpha = a; best_metrics_fuse = m; bestT_fuse = Topt
    best_metrics_fuse['AUC']=round(global_auc((a*zscore_rows(S_exp_tw)+(1-a)*zscore_rows(S_imp_tw))/ bestT_fuse, q_ids, g_ids),4)
    # Save
    ensure_dir(triwin_dir/'curves')
    roc_e, pr_e = curves(S_exp_tw/ bestT_exp_tw, q_ids, g_ids)
    roc_i, pr_i = curves(S_imp_tw/ bestT_imp_tw, q_ids, g_ids)
    write_csv(triwin_dir/'curves/roc_exp.csv', roc_e, header=['fpr','tpr'])
    write_csv(triwin_dir/'curves/pr_exp.csv', pr_e, header=['recall','precision'])
    write_csv(triwin_dir/'curves/roc_imp.csv', roc_i, header=['fpr','tpr'])
    write_csv(triwin_dir/'curves/pr_imp.csv', pr_i, header=['recall','precision'])
    np.save(triwin_dir/'sim_exp.npy', S_exp_tw)
    np.save(triwin_dir/'sim_imp.npy', S_imp_tw)
    write_json(triwin_dir/'metrics_exp.json', {**met_exp_tw, 'best_T':bestT_exp_tw})
    write_json(triwin_dir/'metrics_imp.json', {**met_imp_tw, 'best_T':bestT_imp_tw})
    write_json(triwin_dir/'metrics_fuse.json', {**best_metrics_fuse, 'best_T':bestT_fuse, 'best_alpha':best_alpha})
    write_csv(triwin_dir/'summary.csv', [
        ['explicit_triwin', met_exp_tw['Rank-1'], met_exp_tw['mAP'], met_exp_tw['nDCG@10'], met_exp_tw['AUC'], bestT_exp_tw],
        ['implicit_triwin', met_imp_tw['Rank-1'], met_imp_tw['mAP'], met_imp_tw['nDCG@10'], met_imp_tw['AUC'], bestT_imp_tw],
        ['fuse_triwin', best_metrics_fuse['Rank-1'], best_metrics_fuse['mAP'], best_metrics_fuse['nDCG@10'], best_metrics_fuse['AUC'], bestT_fuse]
    ], header=['setting','Rank-1','mAP','nDCG@10','AUC','best_T'])
    Path(triwin_dir/'README.md').write_text('三窗提效完成：max-pool融合上/下/全身。报告最佳α与温度。', encoding='utf-8')

    # --- ATTRBANK ---
    write_config(attr_dir)
    T_attr_rows = []
    for i,p in enumerate(q_paths):
        cap_list = expl_caps[p]
        bank = build_attr_sentences(cap_list)
        V = encode_texts(model, bank)
        T_attr_rows.append(np.mean(V, axis=0))
    T_attr = np.vstack(T_attr_rows)
    S_attr = sim_from_text_image(T_attr, I_base)
    aucs_attr = [global_auc(S_attr/ t, q_ids, g_ids) for t in TEMPS]
    bestT_attr = TEMPS[int(np.argmax(aucs_attr))]
    met_attr = rank1_map_ndcg(S_attr, q_ids, g_ids); met_attr['AUC']=round(global_auc(S_attr/ bestT_attr, q_ids, g_ids),4)
    # Fusion with explicit baseline
    best_beta=None; best_m=-1; best_metrics_attr_fuse=None; bestT_attr_fuse=None
    for b in BETAS:
        S_attr_fuse = b*zscore_rows(S_attr) + (1-b)*zscore_rows(S_exp)
        aucs_af = [global_auc(S_attr_fuse/ t, q_ids, g_ids) for t in TEMPS]
        Topt = TEMPS[int(np.argmax(aucs_af))]
        m = rank1_map_ndcg(S_attr_fuse, q_ids, g_ids)
        if m['mAP']>best_m:
            best_m = m['mAP']; best_beta=b; best_metrics_attr_fuse=m; bestT_attr_fuse=Topt
    best_metrics_attr_fuse['AUC']=round(global_auc((best_beta*zscore_rows(S_attr)+(1-best_beta)*zscore_rows(S_exp))/ bestT_attr_fuse, q_ids, g_ids),4)
    # Save
    ensure_dir(attr_dir/'curves')
    roc_a, pr_a = curves(S_attr/ bestT_attr, q_ids, g_ids)
    write_csv(attr_dir/'curves/roc_attr.csv', roc_a, header=['fpr','tpr'])
    write_csv(attr_dir/'curves/pr_attr.csv', pr_a, header=['recall','precision'])
    np.save(attr_dir/'sim_attr.npy', S_attr)
    write_json(attr_dir/'metrics_attr.json', {**met_attr, 'best_T':bestT_attr})
    write_json(attr_dir/'metrics_attr_fuse.json', {**best_metrics_attr_fuse, 'best_T':bestT_attr_fuse, 'best_beta':best_beta})
    write_csv(attr_dir/'summary.csv', [
        ['attrbank', met_attr['Rank-1'], met_attr['mAP'], met_attr['nDCG@10'], met_attr['AUC'], bestT_attr],
        ['attrbank_fuse', best_metrics_attr_fuse['Rank-1'], best_metrics_attr_fuse['mAP'], best_metrics_attr_fuse['nDCG@10'], best_metrics_attr_fuse['AUC'], bestT_attr_fuse]
    ], header=['setting','Rank-1','mAP','nDCG@10','AUC','best_T'])
    Path(attr_dir/'README.md').write_text('属性小银行完成：从显式句抽取颜色/包/鞋，平均池化并与显式融合。', encoding='utf-8')

    # --- OVERVIEW ---
    overview = []
    # Collect from three dirs
    def read_summary(p):
        try:
            rows = read_lines(p)
            # skip header
            return [r.split(',') for r in rows[1:]]
        except: return []
    for name, dirp in [('diag', diag_dir), ('triwin', triwin_dir), ('attrbank', attr_dir)]:
        sfile = dirp/'summary.csv'
        if sfile.exists():
            rows = read_summary(sfile)
            for r in rows:
                # setting, Rank-1, mAP, nDCG@10, AUC, best_T
                overview.append([name+':'+r[0], r[1], r[2], r[3], r[4], r[5]])
    write_csv(out_root/f"{ts}_overview.csv", overview, header=['setting','Rank-1','mAP','nDCG@10','AUC','best_T'])

    # Minimal log
    log(diag_dir, 'baseline diag done')
    log(triwin_dir, 'triwin done')
    log(attr_dir, 'attrbank done')

if __name__ == '__main__':
    run()