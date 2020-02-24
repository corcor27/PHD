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

dimesional_array = np.load('2020_02_21_SC_LB151_W800_ST15K_B4K.npy')

                

x = []
y = []
z = []
for k in range(0, LMLO):
    for j in range(0, CC):
        for i in range(0, 700):
            if dimesional_array[k,i,j] > 0:
                x.append(i)
                y.append(j)
                z.append(k)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC)
ax.set_zlim3d(0, LMLO)
ax.view_init(90,80)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC)
ax.set_zlim3d(0, LMLO)
ax.view_init(50,40)