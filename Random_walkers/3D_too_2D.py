import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D

LMLO = 712
CC = 704

multidimensional_array = np.load('2020_02_21_SC_LB151_W800_ST15K_B4K.npy')


LMLO_array = np.zeros((LMLO, 700))
CC_array = np.zeros((CC, 700))
FTF_array = np.zeros((LMLO, CC))

for kk in range(0, LMLO):
    for ii in range(0, 700):
        val = np.sum(multidimensional_array[kk,ii,0:704])
        if val > 30:
            LMLO_array[kk,ii] = val
        
for jj in range(0, CC):
    for ii in range(0, 700):
        val = np.sum(multidimensional_array[0:712,ii,jj])
        if val > 30:
            CC_array[jj,ii] = val
        
for kk in range(0, LMLO):
    for jj in range(0, CC):
        val = np.sum(multidimensional_array[kk,0:700,jj])
        if val > 30:
            FTF_array[kk,jj] = val

plt.imshow(FTF_array, cmap='gray')
plt.colorbar()



