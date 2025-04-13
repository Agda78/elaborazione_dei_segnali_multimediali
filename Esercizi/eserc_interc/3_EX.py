import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from bitop import bitget

def test(x):
    M,N = x.shape

    M_new = 512
    N_new = 512

    # Vado a tagliarmi il triangolo in alto
    new = x[0:M_new, 0:N_new]

    # Calcolo delle feature dell'immagine

    # Estraggo il contributo del secondo bit meno significativo
    B2 = bitget(new,1)

    # Calcolo delle due feature
    h = np.array([[0,1,0],[0,-1,0],[0,0,0]])
    s = ndi.correlate(B2,h)
    
    # Escludo una riga poichè la sommatoria associata parte dalla riga 2
    s = s[1:M_new, :]

    f1 = np.sum(np.abs(s))/((M_new-1)*N_new)

    h = np.array([[0,0,0],[1,-1,0],[0,0,0]])
    s = ndi.correlate(B2,h)

    # Escludo una colonna poichè la sommatoria associata parte dalla colonna 2
    s = s[:,1:N_new]

    f2 = np.sum(np.abs(s))/((N_new-1)*M_new)


    # Calcolo feature 3 e 4
    sigma = 0.5
    y = ndi.gaussian_filter(x, sigma, mode="reflect")

    X = np.fft.fft2(x)
    Y = np.fft.fft2(y)

    f3 = 10*np.log10(np.mean((np.abs(X)-np.abs(Y))**2))
    f4 = 10*np.log10(np.mean((np.angle(X) - np.angle(Y))**2))

    f = 0.7*f1 + 1.5*f2 + 0.01*f3 + 0.001*f4

    return f

if __name__=="__main__":
    num_img = 4

    for i in range(num_img):
        x = io.imread(str(i+1) + ".png")
        x = np.uint8(x)

        f = test(x)

        if f > 1.5 :
            print("[IMG - "+ str(i) +"] from [CAMERA - 1] f = " + str(round(f,3)))
        else:
            print("[IMG - "+ str(i) +"] from [CAMERA - 2] f = " + str(round(f,3)))

    plt.show()