import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D
from tempfile import TemporaryFile


LMLO = 712
CC = 704

dimesional_array = np.load('3D_array.npy')

edge = np.zeros((LMLO,500, CC))
for k in range(0,LMLO):
    for j in range(0, CC):
        for i in range(0,500):
            if dimesional_array[k,i,j] == 1:
                edge[k,i,j] = 1
                break

for k in reversed(range(0,LMLO)):
    for j in reversed(range(0, CC)):
        for i in reversed(range(0,500)):
            if dimesional_array[k,i,j] == 1:
                edge[k,i,j] = 1
                break

np.save('edges_3D_array',edge)
