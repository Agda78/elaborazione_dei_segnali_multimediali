# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# @author: Agda77
# """

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import rgb2gray

def sharpening(x : np.array) -> np.array:
    h = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]])
    flt = ndi.correlate(x, h)
    return flt

def alt_filter(x_arr):
    # Seleziono l'elemento centrale
    x = np.reshape(x_arr, (int(np.sqrt(len(x_arr))),int(np.sqrt(len(x_arr)))))
    # Genero una maschera
    y = (x - x[1,1]) >= 0
    
    sum_u = y[0,0] + 2*y[0,1] + 4*y[0,2] + 8*y[1,2] + 16*y[2,2] + 32*y[2,1] + 64*y[2,0] + 128*y[1,0]
    
    return sum_u   

def true_false(x : np.array) -> bool:
    # 1. Effettuazione del primo processo
    y = sharpening(x)
    
    # 2. Filtro alternativo
    k = 3
    z = ndi.generic_filter(y, alt_filter, size=(k,k))
    
    # 3. Deviazione standard
    
    # std = np.std(z)
    h, b = np.histogram(z,np.arange(257))
    std = np.std(h)
    plt.figure()
    plt.title("Istogramma")
    plt.hist(h.flatten(), bins=255)
    plt.title("var : " + str(std))
    
    
    if std > 495:
        return True
    else:
        return False

if __name__=="__main__":
    
    
    plt.close("all")
    # Main
    x = io.imread("I1.png")
    x = np.float64(x)/255
    x = rgb2gray(x)
    
    y = io.imread("I2.png")
    y = np.float64(y)/255
    y = rgb2gray(y)
    
    message_1 = ""
    if true_false(x) :
        message_1="Vera"
    else:
        message_1="Falsa"
    
    message_2 = ""
    if true_false(y) :
        message_2="Vera"
    else:
        message_2="Falsa"
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=None, cmap="gray")
    plt.title("I1 = " + message_1)
    plt.subplot(1,2,2)
    plt.imshow(y, clim=None, cmap="gray")
    plt.title("I2 = " + message_2)
    
    plt.show()
    