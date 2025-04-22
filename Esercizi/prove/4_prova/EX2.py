import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt

def detect(P1, P2):
    P1_m = ndi.uniform_filter(P1, (127,127))
    P2_m = ndi.uniform_filter(P2, (127,127))
    
    P1_d = P1 - P1_m
    P2_d = P2 - P2_m
    
    num = ndi.generic_filter(P1_d*P2_d, np.sum, (127,127))
    
    a = ndi.generic_filter(P1_d**2, np.sum, (127,127))
    b = ndi.generic_filter(P2_d**2, np.sum, (127,127))
    
    den = np.sqrt(a) * np.sqrt(b)
    
    ro = num/den
    
    ro = ro < 0.02
    
    plt.figure()
    plt.imshow(ro, clim=None, cmap="gray")
    
    return ro
    
    

if __name__=="__main__":
    
    prnu_or = np.load("data_P1.npy")
    prnu_det = np.load("data_P2.npy")
    data_img = np.load("data_img.npy")
    
    
    # plt.figure()
    # plt.imshow(prnu_or, clim=None, cmap="gray")
    # plt.figure()
    # plt.imshow(prnu_det, clim=None, cmap="gray")
    # plt.figure()
    # plt.imshow(data_img, clim=None, cmap="gray")
    
    mask = detect(prnu_or, prnu_det)
    
    nave = np.copy(data_img)
    
    # Identificazione della nave
    nave[:,:,0] = data_img[:,:,0] * mask
    nave[:,:,1] = data_img[:,:,1] * mask
    nave[:,:,2] = data_img[:,:,2] * mask
    
    plt.figure()
    plt.imshow(nave)
    
    plt.show()
