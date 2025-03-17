# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:56:44 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
plt.close('all')

x = io.imread('fragole.jpg')
plt.figure(); plt.imshow(x);

R = x[:,:,0]
plt.figure();
plt.imshow(R, clim=[0,255], cmap='gray');
plt.title('componente di rosso');
G = x[:,:,1]
plt.figure();
plt.imshow(G, clim=[0,255], cmap='gray');
plt.title('componente di verde');
B = x[:,:,2]
plt.figure();
plt.imshow(B, clim=[0,255], cmap='gray');
plt.title('componente di blu');

# M,N,K = x.shape
M = x.shape[0]
N = x.shape[1]

R = np.zeros((M,N), x.dtype) # annullamento della componente di rosso
y = np.stack((R,G,B), -1)
plt.figure();
plt.imshow(y);

