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

dimesional_array = np.load('3D_array.npy')
walkers = 15
steps = []
walker_array = np.zeros((LMLO,700,CC))

es_steps=30000
x0 = int(270)
y0 = int(340)
z0 = int(340)

for ww in range(0, walkers):
    xlist = [x0]
    ylist = [y0]
    zlist = [z0]
    for rr in range(0, es_steps):
        s = np.random.randint(-1,high = 2,size = 1)
        pos = np.random.randint(-1,high = 3,size = 1)
        if pos[0] == 0: 
            xlist.append(s[0])
        elif pos[0] == 1:
            ylist.append(s[0])
        elif pos[0] == 2:
            zlist.append(s[0])
        ii = int(np.sum(xlist))
        jj = int(np.sum(ylist))
        kk = int(np.sum(zlist))
        if dimesional_array[kk,ii,jj] == 0:
            steps.append(rr)
            break
average_steps = round(sum(steps)/4*len(steps))

for ww in range(0, walkers):
    xlist = [x0]
    ylist = [y0]
    zlist = [z0]
    walker_array[kk,ii,jj] = 1
    for rr in range(0, average_steps):
        s = np.random.randint(-1,high = 2,size = 1)
        pos = np.random.randint(-1,high = 3,size = 1)
        if pos[0] == 0: 
            xlist.append(s[0])
        elif pos[0] == 1:
            ylist.append(s[0])
        elif pos[0] == 2:
            zlist.append(s[0])
        ii = int(np.sum(xlist))
        jj = int(np.sum(ylist))
        kk = int(np.sum(zlist))
        walker_array[kk,ii,jj] = 1
        if dimesional_array[kk,ii,jj] == 0:
            break
        
np.save('3D_walker_array4',walker_array)

x = []
y = []
z = []
for k in range(0, LMLO):
    for j in range(0, CC):
        for i in range(0, 500):
            if walker_array[k,i,j] == 1:
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