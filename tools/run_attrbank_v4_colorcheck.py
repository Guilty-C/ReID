import os, json, time
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans

# paths & constants
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

# utils
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

# clip

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

# color detection helpers

def center_square_crop(img):
    w,h=img.size; s=min(w,h); left=(w-s)//2; top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def top_bottom_crops(img):
    sq=center_square_crop(img)
    w,h=sq.size
    top_crop=sq.crop((0,0,w,h//2))
    bot_crop=sq.crop((0,h//2,w,h))
    return top_crop, bot_crop

def kmeans_dominant_rgb(img, k=3):
    arr=np.array(img)/255.0
    if arr.ndim==2: arr=np.stack([arr,arr,arr],axis=-1)
    pixels=arr.reshape(-1,3)
    if len(pixels)>50000:
        idx=np.random.choice(len(pixels), 50000, replace=False)
        pixels=pixels[idx]
    km=KMeans(n_clusters=k, random_state=SEED)
    km.fit(pixels)
    labels=km.labels_
    counts=np.bincount(labels)
    cen=km.cluster_centers_[np.argmax(counts)]
    return tuple([float(x) for x in cen])

def rgb_to_hsv_deg(r,g,b):
    import colorsys
    h,s,v=colorsys.rgb_to_hsv(r,g,b)
    return h*360.0, s, v

def map_rgb_to_11_color(r,g,b):
    h,s,v=rgb_to_hsv_deg(r,g,b)
    if v<0.2: return 'black'
    if s<0.12 and v>0.85: return 'white'
    if s<0.2: return 'gray'
    # hue bands
    if (h>=345 or h<15): return 'red'
    if 15<=h<45: return 'orange' if v>=0.5 else 'brown'
    if 45<=h<75: return 'yellow'
    if 75<=h<165: return 'green'
    if 165<=h<195: return 'teal' if 'green' in COLORS_11 else 'green'
    if 195<=h<255: return 'blue'
    if 255<=h<285: return 'purple'
    if 285<=h<315: return 'pink'
    if 315<=h<345: return 'pink'
    return 'brown'

def predict_top_bottom_colors(img_path):
    try:
        img=Image.open(img_path).convert('RGB')
    except Exception:
        return 'unknown','unknown'
    t,b=top_bottom_crops(img)
    trgb=kmeans_dominant_rgb(t); brgb=kmeans_dominant_rgb(b)
    tc=map_rgb_to_11_color(*trgb); bc=map_rgb_to_11_color(*brgb)
    tc = tc if tc in COLORS_11 else ('green' if tc=='teal' else 'unknown')
    bc = bc if bc in COLORS_11 else ('green' if bc=='teal' else 'unknown')
    return tc, bc

def predict_gallery_colors(model, I_base):
    color_scores={}
    for c,vars in COLOR_VARIANTS.items():
        sents=[f"a person wearing a {v} top" for v in vars]
        V=encode_texts(model,sents); S=V@I_base.T; color_scores[c]=np.max(S,axis=0)
    bottom_scores={}
    for c,vars in COLOR_VARIANTS.items():
        sents=[f"a person wearing {v} pants" for v in vars]
        V=encode_texts(model,sents); S=V@I_base.T; bottom_scores[c]=np.max(S,axis=0)
    G_top=[]; G_bottom=[]
    for j in range(I_base.shape[0]):
        pt=max([(c,color_scores[c][j]) for c in COLORS_11], key=lambda x:x[1])[0]
        pb=max([(c,bottom_scores[c][j]) for c in COLORS_11], key=lambda x:x[1])[0]
        G_top.append(pt); G_bottom.append(pb)
    return G_top, G_bottom

def top1_color_mismatch_ratio(S, q_top, q_bottom, G_top, G_bottom):
    n=S.shape[0]
    mismatches=0
    unknown=0
    for i in range(n):
        j=int(np.argmax(S[i]))
        mt = (q_top[i] != 'unknown' and G_top[j] != q_top[i])
        mb = (q_bottom[i] != 'unknown' and G_bottom[j] != q_bottom[i])
        if (q_top[i]=='unknown' and q_bottom[i]=='unknown'):
            unknown += 1
        if mt or mb:
            mismatches += 1
    return {
        'top1_mismatch': int(mismatches),
        'unknown_queries': int(unknown),
        'total': int(n),
        'ratio': round(mismatches/max(1,n),4)
    }

def image_selfsim_stats(model, preprocess, q_paths):
    sims=[]
    for p in q_paths:
        try:
            img=Image.open(p).convert('RGB')
            top_img, bot_img = top_bottom_crops(img)
            def enc(im):
                x=preprocess(im).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): v=model.encode_image(x)
                v=v/ v.norm(dim=-1, keepdim=True)
                return v.squeeze(0).cpu().numpy()
            v_full = enc(center_square_crop(img))
            v_top = enc(top_img)
            v_bot = enc(bot_img)
            sims.append({
                'file': p,
                'sim_top_full': float(np.dot(v_top, v_full)),
                'sim_bottom_full': float(np.dot(v_bot, v_full)),
                'sim_top_bottom': float(np.dot(v_top, v_bot))
            })
        except Exception:
            sims.append({'file': p, 'sim_top_full': 0.0, 'sim_bottom_full': 0.0, 'sim_top_bottom': 0.0})
    # aggregate
    agg=lambda k: [d[k] for d in sims]
    def summary(arr):
        a=np.array(arr)
        return {
            'mean': float(a.mean()),
            'median': float(np.median(a)),
            'p25': float(np.percentile(a,25)),
            'p75': float(np.percentile(a,75))
        }
    return {
        'sim_top_full': summary(agg('sim_top_full')),
        'sim_bottom_full': summary(agg('sim_bottom_full')),
        'sim_top_bottom': summary(agg('sim_top_bottom')),
        'per_query': sims
    }

# attribute sentence builder (v4 color-checked)

def build_attr_sentences_v4(top_color, bottom_color, bottom_type='pants'):
    s=[]
    if top_color and top_color!='unknown':
        for v in COLOR_VARIANTS[top_color]: s.append(f"a person wearing a {v} top")
    if bottom_color and bottom_color!='unknown' and bottom_type!='unknown':
        for v in COLOR_VARIANTS[bottom_color]: s.append(f"a person wearing {v} {bottom_type}")
    return s[:16]

# main

def main():
    ts_tag=ts(); out_dir=OUT_ROOT/f"{ts_tag}_attrbank_v4_colorcheck"; ensure_dir(out_dir)
    Path(out_dir/'run.log').write_text('',encoding='utf-8')
    # data
    q_paths, g_paths = get_qg(); q_ids=ids_from_paths(q_paths); g_ids=ids_from_paths(g_paths)
    write_csv(out_dir/'pos_per_query.csv', [[basename_noext(q_paths[i]), int(np.sum(np.array(g_ids)==q_ids[i]))] for i in range(len(q_paths))], header=['query','pos_in_gallery'])
    # model & gallery embeds
    model, preprocess = load_clip()
    if EMB_IMAGE_PATH.exists():
        I_base=np.load(EMB_IMAGE_PATH)
        if I_base.shape[0]!=len(g_paths): I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    else:
        ensure_dir(EMB_IMAGE_PATH.parent); I_base=encode_images(model,preprocess,g_paths); np.save(EMB_IMAGE_PATH,I_base)
    # per-query color prediction & sentences (replace if mismatch)
    expl_caps=load_exp_caps(q_paths)
    S_attr_mean_rows=[]; S_attr_max_rows=[]
    q_color_labels={}
    for p in q_paths:
        base=expl_caps[p][0] if expl_caps[p] else 'a person'
        top_pred, bot_pred = predict_top_bottom_colors(p)
        q_color_labels[basename_noext(p)]={'top_color':top_pred,'bottom_color':bot_pred}
        sents_attr=build_attr_sentences_v4(top_pred, bot_pred, bottom_type=('skirt' if 'skirt' in base.lower() else 'pants'))
        if len(sents_attr)==0:
            S_attr_mean_rows.append(np.zeros(len(g_paths))); S_attr_max_rows.append(np.zeros(len(g_paths)))
        else:
            V_attr=encode_texts(model, sents_attr); S_tmp=V_attr@I_base.T
            tops=[i for i,s in enumerate(sents_attr) if ' top' in s]; bots=[i for i,s in enumerate(sents_attr) if ' pants' in s or ' skirt' in s]
            top_sc = np.max(S_tmp[tops], axis=0) if tops else np.zeros(len(g_paths))
            bot_sc = np.max(S_tmp[bots], axis=0) if bots else np.zeros(len(g_paths))
            denom = (1 if tops else 0)+(1 if bots else 0)
            S_attr_mean_rows.append((top_sc+bot_sc)/ (denom if denom>0 else 1))
            S_attr_max_rows.append(np.maximum(top_sc, bot_sc))
    S_attr_mean=np.vstack(S_attr_mean_rows); S_attr_max=np.vstack(S_attr_max_rows)
    np.save(out_dir/'S_attr_mean.npy', S_attr_mean); np.save(out_dir/'S_attr_max.npy', S_attr_max)
    write_json(out_dir/'query_color_labels.json', q_color_labels)
    # metrics & summary
    met_mean=rank1_map_ndcg(S_attr_mean,q_ids,g_ids); met_max=rank1_map_ndcg(S_attr_max,q_ids,g_ids)
    write_json(out_dir/'metrics.json', {'attr_mean':met_mean,'attr_max':met_max})
    write_csv(out_dir/'summary.csv', [
        ['attrbank_v4_colorcheck_mean', met_mean['Rank-1'], met_mean['mAP'], met_mean['nDCG@10'], 'colorcheck','none'],
        ['attrbank_v4_colorcheck_max', met_max['Rank-1'], met_max['mAP'], met_max['nDCG@10'], 'colorcheck','none'],
    ], header=['setting','Rank-1','mAP','nDCG@10','fusion','rerank'])
    # color confusion (using gallery color prediction)
    G_top, G_bottom = predict_gallery_colors(model, I_base)
    conf_mean = top1_color_mismatch_ratio(S_attr_mean, [q_color_labels[basename_noext(p)]['top_color'] for p in q_paths], [q_color_labels[basename_noext(p)]['bottom_color'] for p in q_paths], G_top, G_bottom)
    conf_max = top1_color_mismatch_ratio(S_attr_max, [q_color_labels[basename_noext(p)]['top_color'] for p in q_paths], [q_color_labels[basename_noext(p)]['bottom_color'] for p in q_paths], G_top, G_bottom)
    write_json(out_dir/'color_confusion.json', {'attr_mean':conf_mean,'attr_max':conf_max})
    # self-sim stats
    stats=image_selfsim_stats(model, preprocess, q_paths)
    write_json(out_dir/'img_selfsim_stats.json', stats)
    # config
    write_json(out_dir/'config.yaml', {'filelists':{'query':str(QUERY_LIST),'gallery':str(GALLERY_LIST)},'seed':SEED,'embeds':str(EMB_IMAGE_PATH),'colors':'kmeans_lab_3_to_11'})
    Path(out_dir/'README.md').write_text('v4颜色自检：KMeans(LAB,k=3)取dominant，上/下映射到11色，显式颜色替换为该标签，输出S_attr_*与颜色混淆与自相似诊断。', encoding='utf-8')

if __name__=='__main__':
    main()