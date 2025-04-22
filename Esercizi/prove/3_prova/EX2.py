#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Agda77
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import rgb2gray

if __name__=="__main__":
    # Main
    plt.close("all")
    x = io.imread("ala_ape.jpg")
    x = np.float64(x)/255
    
    x = rgb2gray(x)
    
    
    plt.figure()
    plt.imshow(x, clim=None, cmap="gray")
    plt.title("Originale")
    
    plt.show()