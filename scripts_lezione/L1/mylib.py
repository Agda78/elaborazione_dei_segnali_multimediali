# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:46:55 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def vediJpeg(nomefile):
    x = io.imread(nomefile)
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    
def vediRAW(nomefile, nRighe, nColonne, tipo):
    x = np.fromfile(nomefile, tipo)
    x = np.reshape(x, (nRighe, nColonne))
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    