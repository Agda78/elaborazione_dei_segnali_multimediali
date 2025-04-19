# Librerie
import numpy as np
import skimage.io as io
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
plt.close("all")

def T_opt(x : np.array,k : int):
    centroid, idx, sum_var  = k_means(np.reshape(x, (-1,1)), k)
    t = np.mean(centroid)
    return t

def adaptice_threshold(x, rows):
    d = x[0:rows,:]
    k = 2
    soglia = T_opt(d, k)

    cluster = x > soglia

    return cluster

if __name__=="__main__":
    x = np.fromfile("rice.y", np.uint8)
    x = np.reshape(x, (256,256))

    ideal_mask = np.fromfile("rice_bw.y", np.uint8)
    ideal_mask = np.reshape(ideal_mask, (256,256))

    L_list = []
    n_bit = 8

    for i in range(n_bit + 1):
        L_list.append(2**i)
    
    min_error = 0
    L = 0
    cluster_min = np.array
    for row in L_list:
        cluster = adaptice_threshold(x, row)
        calcolo = ideal_mask - cluster
        error = np.sum(calcolo)

        if (min_error > error) | (row == 1):
            min_error = error
            cluster_min = cluster
            L = row

    plt.figure()
    plt.imshow(x, clim=None, cmap="gray")
    plt.title("Immagine originale")
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Caso Ideale")
    plt.imshow(ideal_mask, clim=[0,1], cmap="jet")
    plt.subplot(1,2,2)
    plt.title("Best,\nerror = " + str(min_error) + ", L = " + str(L))
    plt.imshow(cluster_min, clim=[0,1], cmap="jet")




    # plt.figure()
    # plt.imshow(ideal_mask, clim=None, cmap="gray")

    # plt.figure()
    # plt.imshow(cluster, clim=[0,1], cmap="jet")

    plt.show()