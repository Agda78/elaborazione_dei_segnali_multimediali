import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

from skimage.color import rgb2hsv, hsv2rgb

def mse(x,y):
    mse = np.mean((x - y)**2)
    return mse

plt.close("all")
if __name__=="__main__":
    x = io.imread("pears.png")
    x = np.float32(x)/255

    noisy = io.imread("pears_noise.png")
    noisy = np.float32(noisy)/255

    n_hsv = rgb2hsv(noisy)
    
    # Filtro in frequenza
    V = n_hsv[:,:,2]

    X = np.fft.fftshift(np.fft.fft2(V))

    m = np.fft.fftshift(np.fft.fftfreq(X.shape[0]))
    n = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))

    l,k = np.meshgrid(n,m)

    mu_l = 0.1
    mu_k = 0.1
    raggio = 0.04

    fascia = 0.01
    D1 = np.sqrt((l-mu_l)**2 + (k+mu_k)**2) > raggio
    D2 = np.sqrt((l+mu_l)**2 + (k-mu_k)**2) > raggio
    D3 = ((np.abs(l) - fascia) > mu_l) | ((np.abs(l) + fascia) < mu_l) | (np.abs(l) <= 0.1) | (np.abs(k) <= 0.1)
    D4 = ((np.abs(k) - fascia) > mu_k) | ((np.abs(k) + fascia) < mu_k) | (np.abs(l) <= 0.1) | (np.abs(k) <= 0.1)
    D5 = (np.abs(l + k) >= fascia) | (np.sqrt(k**2 + l**2) <= 0.1)
    D = D1 & D2 & D3 & D4 & D5

    Y = D * X

    V_new = np.real(np.fft.ifft2(np.fft.ifftshift(Y)))

    # Enhancement rendendo più brillante il tutto
    V_new = V_new ** 1.3

    y = hsv2rgb(np.stack((n_hsv[:,:,0], n_hsv[:,:,1], V_new),-1))

    mse_v = mse(x,y)

    plt.figure()
    plt.imshow(x)

    plt.figure()
    plt.imshow(noisy)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(np.log(1 + np.abs(X)), clim=None, cmap="gray", extent=(-0.5,0.5,0.5,-0.5))
    plt.subplot(1,3,2)
    plt.imshow(D, clim=None, cmap="gray", extent=(-0.5,0.5,0.5,-0.5))
    plt.subplot(1,3,3)
    plt.imshow(np.log(1+np.abs(Y)), clim=None, cmap="gray", extent=(-0.5,0.5,0.5,-0.5))

    plt.figure()
    plt.imshow(y)
    plt.title("mse = " + str(mse_v))

    plt.show()