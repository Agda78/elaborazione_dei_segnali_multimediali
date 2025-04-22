import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from mpl_toolkits.mplot3d import Axes3D

# Punto 1
def filtro(M, N, B) -> np.array:
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))

    l,k = np.meshgrid(n,m)

    v_0 = 0.25
    mu_0 = 0.25

    D1 = np.sqrt((l-v_0)**2 + (k-mu_0)**2) < B
    D2 = np.sqrt((l+v_0)**2 + (k+mu_0)**2) < B
    D = D1 | D2
    
    # plt.figure()
    # plt.imshow(D, clim=None, cmap="gray", extent=(-0.5,0.5,0.5,-0.5))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(l, k, D, cmap='viridis')

    return D

# Punto 2
def elabora(x, B):
    x_out = np.copy(x)
    for i in range(3):
        C = x[:,:,i]

        X = np.fft.fftshift(np.fft.fft2(C))
        H = filtro(X.shape[0], X.shape[1],B)

        Y = X*H
        y = np.real(np.fft.ifft2(np.fft.ifftshift(Y)))
        x_out[:,:,i] = y
    
    return x_out

# Calcolo del rapporto segnale rumore
def PSNR(x,y):
    p = 255
    return 10*np.log10(p**2/np.mean((x-y)**2))

if __name__=="__main__":
    x = io.imread("fiori.jpg")
    x = np.float32(x)/255
    B_list = [0.05, 0.10,0.15,0.20]
    psnr_list = []
    for b in B_list:
        flt = elabora(x, b)
        psnr_list.append(PSNR(x,flt))
        print(psnr_list)

    plt.figure()
    plt.plot(B_list, psnr_list)

    plt.show()

