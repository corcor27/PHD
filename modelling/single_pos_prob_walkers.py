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

dimesional_array = np.load('3D_array_test1.npy')
walkers = 100
steps = []
walker_array = np.zeros((LMLO,700,CC))

t = np.amax(dimesional_array)

def prob_value(array, ii, jj, kk):
    lum = dimesional_array[kk,ii,jj]
    p = int(round(100*lum/255))
    return p

for kk in range(0, LMLO):
    for jj in range(0, CC):
        for ii in range(0, 500):
            if dimesional_array[kk,ii,jj] == t:
                x0 = int(ii)
                y0 = int(jj)
                z0 = int(kk)
"""
es_steps=30000


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
        if dimesional_array[kk,ii,jj] <=1:
            steps.append(rr)
            break
average_steps = round(sum(steps)/0.25*len(steps))
"""
average_steps = 30000

for ww in range(0, walkers):
    xlist = [x0]
    ylist = [y0]
    zlist = [z0]
    walker_array[kk,ii,jj] = 1
    for rr in range(0, average_steps):
        a = 1
        while a > 0:
            x_val = np.random.randint(-1,high = 2,size = 1)
            y_val = np.random.randint(-1,high = 2,size = 1)
            z_val = np.random.randint(-1,high = 2,size = 1)
            ii = xlist[rr] + x_val[0]
            jj = ylist[rr] + y_val[0]
            kk = zlist[rr] + z_val[0]
            prob_val = prob_value(dimesional_array, ii, jj, kk)
            s = np.random.randint(-1,high = 101,size = 1)
            if s - prob_val <= 0 and dimesional_array[kk,ii,jj] >=1:
                xlist.append(ii)
                ylist.append(jj)
                zlist.append(kk)
                walker_array[kk,ii,jj] = +1
                a = 0
            
np.save('single_prob_walker',walker_array)
