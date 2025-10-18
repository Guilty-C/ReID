#!/usr/bin/env python
import os, numpy as np
WORKDIR = os.getcwd()
pids = ['P1','P2','P3','P4','P5']
img = os.path.join(WORKDIR,'outputs','embeds','img_iso_real.npy')
I = np.load(img)
assert np.isfinite(I).all(), 'non-finite values in image embeddings'
print('Image shape:', I.shape)
for p in pids:
    T = np.load(os.path.join(WORKDIR,'outputs','embeds',f'text_iso_{p}_norm.npy'))
    assert np.isfinite(T).all(), f'non-finite values in text embeddings {p}'
    ok = (T.shape == I.shape)
    print(f'{p}: Text shape {T.shape} match Image {I.shape} -> {ok}')
print('Done')