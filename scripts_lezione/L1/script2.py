# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:15:37 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

x = io.imread('granelli.jpg'); # leggiamo il file
(M,N) = x.shape # memorizziamo le dimensioni in M e N

#plt.figure(1)
#plt.imshow(x,clim=[0,255], cmap='gray')
#plt.colorbar()

x = np.fromfile('house.y', np.uint8) # lettura dei dati dal file
x = np.reshape(x, (512,512))

x = np.uint8(x) # tipo di dato supportato da JPEG
io.imsave('immagine_q5.jpg', x, quality = 5)

y = io.imread('immagine_q5.jpg')

plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('prima della compressione')
plt.figure(2)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('dopo la compressione') 
