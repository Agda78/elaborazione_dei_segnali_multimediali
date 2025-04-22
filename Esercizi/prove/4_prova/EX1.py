import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from bitop import bitset,bitget

def inserisci_firma(x : np.array, firma : np.array) -> np.array:
    # Rendo la firma puramente binaria
    firma = firma > 0
    
    y = bitset(x, 1, firma)
    
    return y

def mse(x,y):
    return np.mean((x & y)**2)

def confronto_qualita(x, D):
    io.imsave("immagine.jpeg", x, quality = D)
    g = io.imread("immagine.jpeg")
    g = np.uint8(g)
    
    # Firma binaria originale
    firma_or = bitget(x, 1)
    
    # Firma binaria compressa
    firma_comp = bitget(g, 1)
    
    return mse(firma_comp, firma_or)

def filtro_pb(x, D0):
    X = np.fft.fftshift(np.fft.fft2(x))
    m = np.fft.fftshift(np.fft.fftfreq(X.shape[0]))
    n = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))
    
    l,k = np.meshgrid(n,m)
    
    D = np.sqrt(l**2 + k**2) < D0
    
    Y = X * D
    
    y = np.real(np.fft.ifft2(np.fft.ifftshift(Y)))
    y = np.uint8(y)
    
    firma_or = bitget(x, 1)
    firma_flt = bitget(y,1)
    
    return mse(firma_or, firma_flt)
    
if __name__=="__main__":
    
    img = np.fromfile("upupa.y", np.uint8)
    img = np.reshape(img, (256,512))
    
    firma = np.fromfile("firma.y", np.uint8)
    firma = np.reshape(firma, (256,512))
    
    y = inserisci_firma(img, firma)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, clim=None, cmap="gray")
    plt.title("Originale")
    plt.subplot(1,3,2)
    plt.imshow(firma, clim=None, cmap="gray")
    plt.title("Firma")
    plt.subplot(1,3,3)
    plt.imshow(y, clim=None, cmap="gray")
    plt.title("Marcata")
    
    # Robustezza alla compressione
    Q = [80, 90, 100]
    mse_list = []
    for q in Q:
        mse_list.append(confronto_qualita(y, q))
    
    plt.figure()
    plt.plot(Q, mse_list)
    plt.title("Curva MSE Compressione")
    
    # Robustezza alla compressione
    D = [0.2, 0.3, 0.4]
    mse_list = []
    for d in D:
        mse_list.append(filtro_pb(y, d))
    
    plt.figure()
    plt.plot(D, mse_list)
    plt.title("Curva MSE Filtro")
    
    plt.show()