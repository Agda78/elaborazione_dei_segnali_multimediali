# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:50:59 2020

@author: Davide
"""
import numpy as _np

def rgb2hsi(x):
    """
    Conversione spazio di colori da RGB a HSI.
    """
    if x.dtype==_np.uint8:
        x = _np.float64(x)/255 

    # componente I
    I = _np.mean(x, -1)
    
    # componente S
    S = 1.0 - _np.min(x,-1) / _np.maximum(I, _np.finfo(x.dtype).eps)

    # componente H
    den = x @ [1,-0.5,-0.5]
    num = _np.sqrt(3)*(x @ [0,0.5,-0.5])
    H = _np.arctan2(num, den)
    H = (H/(2*_np.pi)) % 1 # normalizzazione componente H

    # ricostruzione
    y = _np.stack((H,S,I),-1)
    return y


def hsi2rgb(x):
    """
    Conversione spazio di colori da HSI a RGB.
    """
    if x.dtype==_np.uint8:
        x = _np.float64(x)/255 

    # estrazione componenti
    cosH = _np.cos(2*_np.pi*x[...,0])
    sinH = _np.sqrt(3)*_np.sin(2*_np.pi*x[...,0])
    P = 3*x[...,0]
    M = 1.0-x[...,1] # minimo relativo
    I = x[...,2]
    
    # inizializzazione
    R = _np.zeros_like(I)
    G = _np.zeros_like(I)
    B = _np.zeros_like(I)
    
    # settore RG, B e' il minimo
    idx = (0 <= P) & (P < 1)
    B[idx] = M[idx]
    R[idx] = ((3-2*B[idx])*cosH[idx] + sinH[idx]) / (cosH[idx]+sinH[idx])
    G[idx] = 3 - (R[idx] + B[idx])
    
    # settore GB, R e' il minimo
    idx = (1 <= P) & (P < 2)
    R[idx] = M[idx]
    G[idx] = 3/2 + (R[idx]-1.0)*sinH[idx]/cosH[idx] /2   - R[idx]/2
    B[idx] = 3 - (R[idx] + G[idx])
    
    # settore BR, G e' il minimo
    idx = (2 <= P) & (P <= 3)
    G[idx] = M[idx]
    R[idx] = ((3-2*G[idx])*cosH[idx] - sinH[idx]) / (cosH[idx]-sinH[idx])
    B[idx] = 3 - (G[idx] + R[idx])

    # ricostruzione immagine RGB
    y = I[...,None] * _np.stack((R,G,B), -1)
    return y