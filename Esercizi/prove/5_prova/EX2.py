import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import rgb2gray
import skimage.morphology as morph

if __name__=="__main__":
    plt.close("all")
    x = io.imread("cells.png")
    x = np.float64(x)
    x = x[:,:,0]
    h,b = np.histogram(x, np.arange(257))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('immagine')
    plt.subplot(1,2,2)
    plt.hist(x.flatten(), bins=256)
    plt.ylim([0,1000])
    plt.title('istogramma')
    # vederndo l'istogramma si evidenziono due soglie a 51 e 155
    
    mask = x>51 # maschera di tutte le cellule
    plt.figure()
    plt.imshow(mask, clim=None, cmap="gray")
    
    b = morph.disk(5)
    mask = morph.binary_opening(mask, b)
    plt.figure()
    plt.imshow(mask, clim=None, cmap="gray")
    
    b = morph.disk(2)
    bordi = mask ^ morph.binary_erosion(mask, b)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mask, clim=[0,1], cmap='gray')
    plt.title('tutte le cellule')
    plt.subplot(1,2,2)
    plt.imshow(bordi, clim=[0,1], cmap='gray')
    plt.title('bordi')
    
    chiare = x>170
    plt.figure()
    plt.imshow(chiare, clim=None, cmap="gray")
    b = morph.disk(2)
    chiare = morph.binary_opening(chiare, b)
    # prende solo la parte centrale delle chaire
    
    
    # dilation iterativa per prendere l'intre cellula
    b = morph.disk(1)
    for i in range(100):
        chiare = morph.binary_dilation(chiare, b) & mask
    
    scure = mask ^ chiare # Xor
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(scure, clim=[0,1], cmap='gray')
    plt.title('scure')
    plt.subplot(1,2,2)
    plt.imshow(chiare, clim=[0,1], cmap='gray')
    plt.title('chiare')
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(x, clim=None, cmap="gray")
    # plt.title("Originale")
    # plt.subplot(1,2,2)
    # plt.hist(x.flatten(), bins=256)
    
    
    
    