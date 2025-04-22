import numpy as np
import skimage.io as io
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def copia_qualita(x : np.array):
    # Q = 1:10:100 visto come comando matlab
    Q = np.arange(start=0,stop=101, step=10)
    Q[0] = 1
    
    for q in Q:
        # Generazione immagine di qualità diverse
        
        # 1. Vado a considerare l'immagine a vari livelli di qualità
        io.imsave("immagine.jpeg", x, quality=q)
        y = io.imread("immagine.jpeg")
        y = np.float64(y)/255
        
        # 2. Calcolo di d_q
        d_q = (x - y)**2
        
        # 3. Compattazione singola banda e filtro spaziale
        dqg = (d_q[:,:,0] + d_q[:,:,1] + d_q[:,:,2])/3
        
        # filtro
        flt = ndi.uniform_filter(dqg, (16,16))
        
        plt.figure()
        plt.imshow(flt,clim=None, cmap="gray")
        plt.title("Qualita = " + str(q))

if __name__=="__main__":
    plt.close("all")
    x = io.imread("auto.jpg")
    x = np.float32(x)/255
    
    # copia_qualita(x)
    
    # Visti i grafici mi conviene quello con qualità 90
    io.imsave("immagine.jpeg", x, quality=90)
    y = io.imread("immagine.jpeg")
    y = np.float64(y)/255
    
    # 2. Calcolo di d_q
    d_q = (x - y)**2
    
    # 3. Compattazione singola banda e filtro spaziale
    dqg = (d_q[:,:,0] + d_q[:,:,1] + d_q[:,:,2])/3
    
    # filtro
    flt = ndi.uniform_filter(dqg, (16,16))
    
    th=1.5e-5
    
    mask = flt < th
    
    plt.figure()
    plt.imshow(mask,clim=None, cmap="gray")
    plt.title("Qualita = " + str(q))
    
    plt.figure()
    plt.imshow(x, clim=None, cmap="gray")
    plt.title("Originale")
    
    plt.show()

