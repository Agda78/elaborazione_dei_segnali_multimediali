import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import rgb2gray

def vedi_laplaciano():
    h = np.array([[1,0,0,0,1],[0,0,0,0,0],[0,0,-4,0,0],[0,0,0,0,0],[1,0,0,0,1]])
    H = np.fft.fftshift(np.fft.fft2(h))
    
    plt.figure()
    plt.imshow(np.log(1+np.abs(H)), clim=None, cmap="gray", extent=(-0.5,0.5,0.5,-0.5))
    plt.title("Filtro Laplaciano")

def space_filt(x : np.array) -> np.array:
    h = np.array([[1,0,0,0,1],[0,0,0,0,0],[0,0,-4,0,0],[0,0,0,0,0],[1,0,0,0,1]])
    
    y = ndi.convolve(x,h)
    
    return y
    
def freq_filt(x : np.array) -> np.array:
    h = np.array([[1,0,0,0,1],[0,0,0,0,0],[0,0,-4,0,0],[0,0,0,0,0],[1,0,0,0,1]])
    
    M,N = x.shape
    A,B = h.shape
    
    P = M+A-1
    Q = N+B-1
    
    X = np.fft.fft2(x, (P,Q))
    H = np.fft.fft2(h, (P,Q))
    
    Y = X*H
    y = np.real(np.fft.ifft2(Y))
    return y[:M,:N] 
    
def mse(x,y):
    return np.mean((x-y)**2)
    
if __name__ == "__main__":
    plt.close("all")
    
    x = io.imread("barbara.jpg")
    x = np.float32(x)/255
    x = rgb2gray(x)
    
    
    # 1.Vedere il filtro laplaciano
    vedi_laplaciano()
    
    # 2. Filtraggio nello spazio
    y1 = space_filt(x)
    
    # 3. Filtraggio nella frequenza
    y2 = freq_filt(x)
    
    print(mse(y1,y2))
    plt.figure()
    plt.imshow(x, clim=None, cmap="gray")
    plt.title("Originale\nmse= " + str(mse(y1,y2)))
    plt.show()
