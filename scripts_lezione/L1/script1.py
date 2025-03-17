# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:52:23 2025

@author: Davide
"""

import numpy as np

V = np.array([1,2,3,4,5,6])

M = np.array([[1,2,3],[4,5,6] ])

M2 = np.float32(M)

A = np.copy(V)
A[1] = 0

P = np.arange(5,10,3)

b = M[:,::2]

min_M = np.min(M)

min_row = np.min(M,0)
min_col = np.min(M,1)
# commento
"""
commenti
"""

np.save('array.npy', M)

M3 = np.load('array.npy')


