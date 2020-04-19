import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D
import random



LMLO = 712
CC = 704

dimesional_array = np.load('3D_array_test1.npy')
walkers = 100
steps = []
walker_array = np.zeros((LMLO,700,CC))

t = np.amax(dimesional_array)

print(t)

for kk in range(0, LMLO):
    for jj in range(0, CC):
        for ii in range(0, 500):
            if dimesional_array[kk,ii,jj] == t:
                x0 = int(ii)
                y0 = int(jj)
                z0 = int(kk)

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
average_steps = 20000



def position_check(x,z,y,array):
    if array[z,x,y] > 0:
        return 0
    else:
        return 1
    
def positions_proposed(i,k,j,array):
# functions creates array of new positions
    new_x = []
    new_z = []
    new_y = []
    for di in range(-1,2):
        for dk in range(-1,2):
            for dj in range(-1,2):
                x = int(i+di)
                z = int(k+dk)
                y = int(j+dj)
                new_x.append(x)
                new_z.append(z)
                new_y.append(y)
    return new_x, new_y, new_z

def probability_array(array,map_array, rr):
    x = []
    for yy in range(0, len(array[0])):
        ii = array[0][yy]
        jj = array[1][yy]
        kk = array[2][yy]
        lum = map_array[kk,ii,jj]
        w = round(lum/(rr+1), 2)
        x.append(w)
    return x
        
    
for ww in range(0, walkers):
    xlist = [x0]
    ylist = [y0]
    zlist = [z0]
    
    for rr in range(0, average_steps):
         
        
        s = random.randint(1,99)
        i = xlist[rr]
        j = ylist[rr]
        k = zlist[rr]
        walker_array[k,i,j] = 1
        av_steps = positions_proposed(i,k,j,walker_array)
        prob_w = np.asarray(probability_array(av_steps, dimesional_array, rr))
        total_w = sum(prob_w)
        inter = round(100/total_w, 4)
        
        p_val_array = inter*prob_w
        check = np.sum(p_val_array)
        for tt in range(0, len(p_val_array)):
            s = s - p_val_array[tt]
            if s <= 0:
                xlist.append(av_steps[0][tt])
                ylist.append(av_steps[1][tt])
                zlist.append(av_steps[2][tt])
                walker_array[av_steps[2][tt],av_steps[0][tt],av_steps[1][tt]] = 1
                break
        
        
        

        
np.save('3D_prob_walker_array1',walker_array)

