#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Agda77
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import rgb2gray, gray2rgb

def laplaciano(x : np.array) -> np.array:
    h = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    
    lpc = ndi.correlate(x,h)
    
    return lpc ** 2

def livello_attivita(x : np.array, k : int) -> np.array:
    
    var_loc = ndi.generic_filter(x, np.var, size=(k,k))
    avg_loc = ndi.generic_filter(x, np.mean, size=(k,k))
    
    return avg_loc * var_loc


if __name__=="__main__":
    plt.close("all")
    
    x_1 = io.imread("disk1.jpg")
    x_1 = np.float32(x_1)/255
    x_1 = rgb2gray(x_1)
    
    x_2 = io.imread("disk2.jpg")
    x_2 = np.float32(x_2)/255
    x_2 = rgb2gray(x_2)
    
    # 1. Calcolo del quadrato dei laplaciani
    lpc_1 = laplaciano(x_1)
    lpc_2 = laplaciano(x_2)
    
    # 2. Calcolo del livello di attività
    k = 5 # Dimensione della finestra
    a_1 = livello_attivita(x_1, k)
    a_2 = livello_attivita(x_2, k)
    
    # 3. Normalizzazione
    a_1 = a_1/(a_1+a_2+1e-15)
    a_2 = 1-a_1
    
    # Fusione
    x_f = (a_1 * x_1) + (a_2 * x_2)
    
    
    #x_1 = gray2rgb(x_1)
    #x_2 = gray2rgb(x_2)
    #x_f = gray2rgb(x_f)
    
    plt.figure()
    
    plt.subplot(1,3,1)
    plt.imshow(x_1, clim=None, cmap="gray")
    plt.title("Disk 1")
    plt.subplot(1,3,2)
    plt.imshow(x_2, clim=None, cmap="gray")
    plt.title("Disk 2")
    plt.subplot(1,3,3)
    plt.imshow(x_f, clim=None, cmap="gray")
    plt.title("Filtrata")
    
    plt.show()