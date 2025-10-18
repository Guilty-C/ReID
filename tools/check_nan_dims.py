import os, numpy as np
workdir = os.getcwd()
T = np.load(os.path.join(workdir, 'outputs', 'embeds', 'text_iso_real.npy'))
I = np.load(os.path.join(workdir, 'outputs', 'embeds', 'img_iso_real.npy'))
print('text shape:', T.shape, 'nan:', np.isnan(T).sum())
print('image shape:', I.shape, 'nan:', np.isnan(I).sum())