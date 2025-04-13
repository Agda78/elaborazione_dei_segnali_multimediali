# Import delle varie librerie
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def media_adattiva(x : np.array, T):
    k = int(np.sqrt(len(x)))
    x = np.reshape(x, (k,k))

    if np.var(x) < T:
        return np.mean(x)
    else:
        return np.mean(x[1:k-1,1:k-1])

def adapt_filter(x, k):
    # 1. Calcolo le varianze locali
    # var_loc = ndi.generic_filter(x, np.var, (k,k), mode="reflect")

    # 2. Definisco il valore di soglia
    x_flat = x.flatten()
    x_flat = np.sort(x_flat)
    T = x_flat[int(0.7*len(x_flat))]

    # 3. Vado a filtrare ogni blocco k,k con la funzione ad-hoc
    y = ndi.generic_filter(x, media_adattiva, (k,k), extra_keywords={"T":T})

    return y

def PSNR(x, y):
    p = 255.0  # Massimo valore rappresentabile
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')  # Le immagini sono identiche
    psnr = 10 * np.log10(p ** 2 / mse)
    return psnr

def add_noise(x, std):
    # Genero il rumore
    M,N = x.shape
    n = std*np.random.randn(M,N)

    # Calcolo l'immagine rumorosa
    noisy = x + n

    return noisy

if __name__=="__main__":
   
    # Carico l'immagine
    x = io.imread("cigno.jpg")
    x = np.float32(x)

    std = 25
    noisy = add_noise(x, std)
    
    # Visualizzazione immagine originale + rumorosa
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255],cmap="gray")
    plt.title("Originale")

    plt.subplot(1,2,2)
    plt.imshow(noisy, clim=[0,255], cmap="gray")
    plt.title("Rumorosa")

    # RISOLUZIONE

    k_list = [3, 5, 7, 9]
    adapt_psnr = []
    media_psnr = []

    for i in k_list:
        y = adapt_filter(noisy, i)

        # Definire le ultime operazioni
        adapt_psnr.append(PSNR(x,y))

        y_avg = ndi.uniform_filter(noisy, (i,i))

        media_psnr.append(PSNR(x,y_avg))
        
        if (i==5) :
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(y, clim=[0,255], cmap="gray")
            plt.title("Media Adattiva")

            plt.subplot(1,2,2)
            plt.imshow(y_avg, clim=[0,255],cmap="gray")
            plt.title("Media Uniforme")

    plt.figure()
    plt.plot(k_list, adapt_psnr, label="Media Adattiva")
    plt.plot(k_list, media_psnr, label="Media Uniforme")
    plt.title("Confronto PSNR")
    plt.legend()

    plt.show()