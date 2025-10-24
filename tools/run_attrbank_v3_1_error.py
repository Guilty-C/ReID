import os, json, time
from pathlib import Path
import numpy as np
import torch, clip

ROOT=Path('d:/PRP SunnyLab/ReID')
OUT_ROOT=ROOT/'outputs/comparison'
EMB_IMAGE_PATH = ROOT/'embeds/image/clip-l14_market_subset50.npy'
# prefer subset100 lists if available
QUERY_LIST = ROOT/'larger_iso/64/query100.txt' if (ROOT/'larger_iso/64/query100.txt').exists() else ROOT/'larger_iso/64/query.txt'
GALLERY_LIST = ROOT/'larger_iso/64/gallery100.txt' if (ROOT/'larger_iso/64/gallery100.txt').exists() else ROOT/'larger_iso/64/gallery.txt'
EXPL_CAP_PATH = ROOT/'outputs/captions/explicit_captions.json'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
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

def ts(): return time.strftime('%Y%m%d-%H%M%S')
read_lines=lambda p: [s.strip() for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
basename_noext=lambda p: os.path.splitext(os.path.basename(p))[0]

def ids_from_paths(paths):
    ids=[]
    for p in paths:
        s=basename_noext(p); head=s.split('_')[0]
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

def normalize_color(text):
    t=text.lower()
    for c in COLORS_11:
        if c in t: return c
    for c,vars in COLOR_VARIANTS.items():
        for v in vars:
            if v in t: return c
    return 'unknown'

def load_clip():
    model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    return model, preprocess

def encode_texts(model, texts):
    import clip as cl
    toks=cl.tokenize(texts).to(DEVICE)
    with torch.no_grad(): v=model.encode_text(toks)
    v=v/ v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()

def main():
    # locate latest v3.1_lr dir
    cand=sorted([p for p in OUT_ROOT.glob('*_attrbank_v3_1_lr')])
    if not cand: print('No v3.1_lr found'); return
    lr_dir=cand[-1]
    # base error dir
    err_dir = OUT_ROOT/f"{lr_dir.name.replace('_attrbank_v3_1_lr','')}_attrbank_v3_1_error"; err_dir.mkdir(parents=True, exist_ok=True)
    # data
    q_paths, g_paths = read_lines(QUERY_LIST), read_lines(GALLERY_LIST)
    q_ids, g_ids = ids_from_paths(q_paths), ids_from_paths(g_paths)
    S_lr = np.load(lr_dir/'S_fuse_lr.npy')
    # gallery embeds (for color prediction per gallery)
    I_base = np.load(EMB_IMAGE_PATH)
    model,_ = load_clip()
    # precompute gallery color scores by max-over-variants
    color_scores={}
    for c,vars in COLOR_VARIANTS.items():
        sents=[f"a person wearing a {v} top" for v in vars]
        V=encode_texts(model, sents)
        S=V @ I_base.T  # n_var x G
        color_scores[c]=np.max(S, axis=0)  # G
    # parse query top color
    expl=load_exp_caps(q_paths)
    q_top=[normalize_color(expl[p][0]) for p in q_paths]
    # collect top-20 failures
    rows=[]; cause_cnt={}
    for i in range(S_lr.shape[0]):
        order=np.argsort(-S_lr[i])
        wrong=[j for j in order if g_ids[j]!=q_ids[i]][:20]
        for j in wrong:
            # predicted gallery top color
            preds=[(c,color_scores[c][j]) for c in COLORS_11]
            pred=max(preds, key=lambda x:x[1])[0]
            cause = '颜色混淆' if (q_top[i]!='unknown' and pred!=q_top[i]) else '其他'
            rows.append([q_ids[i], g_ids[j], cause])
            cause_cnt[cause]=cause_cnt.get(cause,0)+1
    # write
    def write_csv(p,rows,header=None):
        txt=[]
        if header: txt.append(','.join(header))
        for r in rows: txt.append(','.join(str(x) for x in r))
        Path(p).write_text('\n'.join(txt), encoding='utf-8')
    write_csv(err_dir/'top20_failures.csv', rows, header=['query_id','gallery_id','cause'])
    Path(err_dir/'README.md').write_text(f"top-20失败统计：颜色混淆={cause_cnt.get('颜色混淆',0)}，其他={cause_cnt.get('其他',0)}", encoding='utf-8')
    # delta vs previous v3 error
    prev_err=sorted([p for p in OUT_ROOT.glob('*_attrbank_v3_error')])
    delta_rows=[['cause','v3','v3.1','delta']]
    if prev_err:
        prev=prev_err[-1]
        prev_rows = Path(prev/'top20_failures.csv').read_text(encoding='utf-8').splitlines()[1:]
        pc={}
        for ln in prev_rows:
            c=ln.split(',')[-1]; pc[c]=pc.get(c,0)+1
        causes=set(list(pc.keys())+list(cause_cnt.keys()))
        for c in causes:
            v3=pc.get(c,0); v31=cause_cnt.get(c,0); delta_rows.append([c,v3,v31,v31-v3])
    else:
        for c,v in cause_cnt.items(): delta_rows.append([c,0,v,v])
    write_csv(err_dir/'failures_delta.csv', delta_rows)

if __name__=='__main__':
    main()