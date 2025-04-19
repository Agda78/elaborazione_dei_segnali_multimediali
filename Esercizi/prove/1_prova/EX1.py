# Librerie
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

import skimage.util as util

plt.close("all")

def mask_fun(x, T):
    # Calcolo la dimensione k della finestra
    dim = int(np.sqrt(len(x)))

    # Calcolo il valore mediano
    x_sort = np.sort(x)
    m = x_sort[int(len(x)/2)]

    # Riprendo il valore centrale
    x_cent = x[int(len(x)/2)]

    if np.abs(m - x_cent) > T:
        # Z=1 il pixel dev'essere elaborato
        return m
    else:
        # Z=0 il pixel non dev'essere elaborato
        return x_cent

def smf(x : np.array, k : int, T : float) -> np.array:
    # Funzione di generazione della maschera su cui agire
    result = ndi.generic_filter(x, mask_fun, (k,k), mode="reflect", extra_keywords={"T":T})
    return result

def psnr(x : np.array, y : np.array) -> float:
    p = 255 # Valore massimo possibile nell'immagine
    psnr = 10 * np.log10(p**2 / MSE(x,y))
    return psnr

def MSE(x : np.array, y : np.array) -> float:
    mse = np.mean((x - y)**2)
    return mse


if __name__ == "__main__":
    x = io.imread("lena.jpg")
    x = np.uint8(x)

    p = 0.2
    y = util.random_noise(x, mode="s&p", amount=p)
    y = y * 255

    # Visualizzazione delle immagini effettive
    plt.figure("1.Immagini Iniziali")
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap="gray")
    plt.title("Originale")
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap="gray")
    plt.title("Rumorosa")

    plt.figure("Istogramma Rumorosa")
    plt.hist(y.flatten(), bins=256)
    k_list = [3,5,7,9,11]
    T = 22

    adapt_psnr = []
    median_psnr = []

    for k in k_list: 
        z = smf(x, k, T)
        pn = psnr(x, z)
        adapt_psnr.append(pn)

        z_med = ndi.median_filter(x, (k,k), mode="reflect")
        pn = psnr(x, z_med)
        median_psnr.append(pn)

        if k == 5:
            plt.figure()
            plt.subplot(1,2,1)
            plt.title("Classic Median")
            plt.imshow(z_med, clim=None, cmap="gray")
            plt.subplot(1,2,2)
            plt.title("Adaptive median")
            plt.imshow(z, clim=None, cmap="gray")

    plt.figure()
    plt.plot(k_list, adapt_psnr, label="Adaptive")
    plt.plot(k_list, median_psnr, label="Median")

    plt.legend()

    plt.show()