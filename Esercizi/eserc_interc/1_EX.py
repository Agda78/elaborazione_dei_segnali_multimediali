import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

def fshs(x, K):
    x_max = np.max(x)
    x_min = np.min(x)

    y = (K-1)*(x-x_min)/(x_max - x_min)

    return y

def median_filter(x, k):
    y = ndi.median_filter(x, (k,k), mode="reflect")
    return y

def dettagli(x):
    h = np.array([[0,0,0],[0,1,-1],[0,0,0]])
    
    y = ndi.correlate(x,h)

    return y

if __name__=="__main__":
    
    # Carico l'immagine
    x = io.imread("ponte.png")
    x = np.float32(x)

    # Visualizzo:
    # - Originale
    # - Istogramma (con massimi e minimi)
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap="gray")
    plt.title("Originale")

    plt.subplot(1,2,2)
    plt.hist(x.flatten(), bins=256)
    plt.title("max = " + str(np.max(x)) + "\nmin = " + str(np.min(x)))


    # Dimensione della finestra del median_filter
    k = 3
    y = median_filter(x, k)

    # Vado a stratchiare l'istogramma perchè è troppo concetrato
    z = fshs(y, 256)

    # Qui bisogna effettuare delle operazioni per migliorare i bordi
    # - eventuali filtri di sharpening
    # - laplaciano per migliorare i bordi

    # Visualizzazione dei risultati ottenuti dai vari processi
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(y, clim=[0,255], cmap="gray")
    plt.title("Filtro Mediano")

    plt.subplot(1,3,2)
    plt.imshow(z, clim=[0,255],cmap="gray")
    plt.title("Filtro Mediano + FSHS")

    plt.show()