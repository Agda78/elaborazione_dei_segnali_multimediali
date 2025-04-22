import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from skimage.color import gray2rgb, rgb2gray, rgb2hsv, hsv2rgb

def edge_detection(x : np.array) -> np.array:
    """
    Ritorna l'immagine elaborata secondo una tecnica di edge detection
    """
    # Fase 1: Derivata direzionale dell'immagine
    h = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    y = ndi.correlate(x,h)

    # Fase 2: Threshholding
    soglia = np.max(y)*0.10
    y = y > soglia

    return y

def alternative_filter(x, T):
    """
    Funzione da utilizzare su blocchi limitati di immagine
    """
    avg = np.mean(x**2)
    avg_gem = (np.prod(x**2)**(1/len(x)))

    if avg_gem != 0:
        Rag = avg/avg_gem
    else:
        Rag = avg

    if Rag >= T:
        return 1
    else:
        return 0
    
def alternative_detection(x : np.array, k : int, T : float) -> np.array:
    """
    Sapendo che la x sarà in scala di grigi
    """
    
    y = ndi.generic_filter(x, alternative_filter, (k,k), mode="reflect", extra_keywords={"T":T})

    return y

if __name__=="__main__":
    x = np.fromfile("target_rumorosa.raw", np.float32)/255
    x = np.reshape(x, (256,256))

    y = edge_detection(x)

    k = 7
    T = np.mean(x**2)
    z = alternative_detection(x, k, T)

    plt.figure("1.Originale")
    plt.imshow(x, clim=None, cmap="gray")
    
    plt.figure("2.Edge_detection")
    plt.imshow(y, clim=None, cmap="gray")

    plt.figure("3.Alternative Detection")
    plt.imshow(z, clim=None, cmap="gray")
    
    plt.show()