import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from color_convertion import rgb2hsi, hsi2rgb

if __name__=="__main__":
    # Main
    x = io.imread("foto_originale.tif")
    x = np.float32(x)/255

    plt.figure()
    plt.title("Immagine Originale")
    plt.imshow(x)

    # Converto l'immagine in HSI, in modo da andare ad agire solo sulle tinte I
    x_hsi = rgb2hsi(x)

    H = x[:,:,0]
    S = x[:,:,1]
    I = x[:,:,2]

    X = np.fft.fftshift(np.fft.fft2(I))

    M,N = I.shape

    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))

    l,k = np.meshgrid(n,m)

    mask = (np.abs(l) <= 0.10) & (np.abs(k) <= 0.25)

    X = X * mask

    I = np.real(np.fft.ifft2(np.fft.ifftshift(X)))

    x_filtered = np.stack((H,S,I),-1)

    x_filtered = hsi2rgb(x_filtered)

    plt.figure()
    plt.title("Immagine Filtrata")
    plt.imshow(x_filtered)


    plt.show()