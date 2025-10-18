import os
from PIL import Image, ImageDraw, ImageFont
import hashlib
import random

ROOT = os.path.join(os.getcwd(), 'data', 'mock_market')
Q = os.path.join(ROOT, 'query')
G = os.path.join(ROOT, 'bounding_box_test')

os.makedirs(Q, exist_ok=True)
os.makedirs(G, exist_ok=True)

W, H = 160, 320

def color_for(name):
    h = hashlib.sha256(name.encode('utf-8')).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)

def make_img(path):
    name = os.path.basename(path)
    bg = color_for(name)
    img = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(img)
    # simple person silhouette: head + body rectangle
    cx, cy = W//2, H//4
    d.ellipse((cx-12, cy-12, cx+12, cy+12), fill=(255,255,255))
    d.rectangle((cx-20, cy+12, cx+20, cy+80), fill=(240,240,240))
    d.rectangle((cx-30, cy+80, cx+30, cy+160), outline=(230,230,230), width=2)
    # label
    try:
        d.text((6, H-18), name, fill=(0,0,0))
    except Exception:
        pass
    img.save(path, format='JPEG', quality=85)

fixed = 0
for sub in (Q, G):
    for fn in os.listdir(sub):
        if not fn.lower().endswith('.jpg'):
            continue
        p = os.path.join(sub, fn)
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        if size == 0:
            make_img(p)
            fixed += 1

print(f"[OK] fixed {fixed} zero-byte images")