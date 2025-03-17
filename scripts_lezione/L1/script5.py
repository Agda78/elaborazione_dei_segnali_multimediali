# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:11:14 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
plt.close('all')

x = io.imread('dorian.jpg'); # leggiamo il file
x = np.float64(x)

xmed = np.mean(x) # media
xstd = np.std(x) # deviazione standard
var = np.var(x) # varianza = xstd **2
plt.figure(); plt.imshow(x, clim=[0,255], cmap='gray');

M,N = x.shape
MED = np.zeros((M-2,N-2))
for i in range(M-2):
    for j in range(N-2):
        MED[i,j] = np.mean(x[i:i+3,j:j+3])
        
plt.figure(); plt.imshow(MED, clim=[0,255], cmap='gray');

y = ndi.generic_filter(x, np.var, (3,3))
plt.figure();
plt.subplot(1,2,1); plt.imshow(x, clim=[0,255], cmap='gray');
plt.subplot(1,2,2); plt.imshow(y, clim=None, cmap='gray');

