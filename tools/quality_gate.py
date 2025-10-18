
import json,sys,argparse
ap=argparse.ArgumentParser()
ap.add_argument('baseline'); ap.add_argument('current')
ap.add_argument('--delta',type=float,default=0.0)
a=ap.parse_args()
b=json.load(open(a.baseline)); c=json.load(open(a.current))
ok=(c['mAP']>=b['mAP']-a.delta) and (c['rank1']>=b['rank1']-a.delta)
print('[GATE]', 'PASS' if ok else 'FAIL', {'mAP':c['mAP'],'rank1':c['rank1']}, '>=?', {'mAP':b['mAP']-a.delta,'rank1':b['rank1']-a.delta})
sys.exit(0 if ok else 2)
