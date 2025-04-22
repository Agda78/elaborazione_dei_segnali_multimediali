import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
import skimage.morphology as morph


if __name__ == "__main__":
    plt.close("all")
    x = io.imread("img1.png")
    x = np.float64(x)
    
    y = io.imread("img2.png")
    y = np.float64(y)
    
    plt.figure()
    plt.imshow(x, clim=None, cmap="gray")
    
    plt.figure()
    plt.imshow(y, clim=None, cmap="gray")
    raggio = 18
    c = morph.disk(18)
    
    g1 = ndi.gaussian_filter(x, (10,10)) < 15
    g2 = ndi.gaussian_filter(y, (10,10)) < 15
    
    plt.figure()
    plt.imshow(g1, clim=None, cmap="gray")
    plt.figure()
    plt.imshow(g2, clim=None, cmap="gray")