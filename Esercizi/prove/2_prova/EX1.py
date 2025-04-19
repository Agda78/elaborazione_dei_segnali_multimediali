# Librerie
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import rgb2gray

def adapt_filter(x, sigma):
    corrente = x[int(len(x)/2)]
    mask = np.abs(x) < corrente + 2*sigma
    num_el = np.sum(mask)
    
    if num_el < 4 :
        # Media normale
        return np.mean(x)
    else:
        # Media solo sugli elementi considerati
        x = mask * x
        print(np.sum(mask))
        return np.sum(x)/num_el

def filtro_sigma(x, k, sigma):
    flt = ndi.generic_filter(x, adapt_filter, (k,k), extra_keywords={"sigma":sigma})
    return flt

def psnr(x,y):
    p = 255
    return 10*np.log10(p**2/mse(x,y))

def mse(x,y):
    return np.mean((x-y)**2)

if __name__=="__main__":
    # Main
    x = np.float64(io.imread("barbara.jpg"))
    x = rgb2gray(x)

    (M,N) = x.shape
    std_noise = 20
    n = std_noise*np.random.randn(M,N)

    noisy = x + n

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap="gray")
    plt.title("Originale")
    plt.subplot(1,2,2)
    plt.imshow(noisy, clim=[0,255], cmap="gray")
    plt.title("Rumorosa")


    k = 7
    flt = filtro_sigma(noisy, k, std_noise)

    psnr_v = psnr(x, flt)

    plt.figure()
    plt.imshow(flt, clim=[0,255], cmap="gray")
    plt.title("Filtrata\nPSNR = " + str(round(psnr_v,3)))


    plt.show()