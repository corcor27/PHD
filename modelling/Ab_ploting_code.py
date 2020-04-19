import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D
import os

def graph_array(addition1):
    dimesional_array = np.load(addition1)
    LMLO = dimesional_array.shape[0]
    CC = dimesional_array.shape[1]
    FTF = dimesional_array.shape[2]
    count = 0
    for k in range(0, LMLO):
        for j in range(0, CC):
            for i in range(0, FTF):
                if dimesional_array[k,i,j] > 1001:
                    count += 1
                
    print(count)
    positions_array = np.zeros((count, 3))
    count1 = 0
    for k in range(0, LMLO):
        for j in range(0, CC):
            for i in range(0, FTF):
                if dimesional_array[k,i,j] > 1001:
                    positions_array[count1, :] = i, j, k
                    count1 += 1
                    
                        
                    
    return positions_array
    

path =r"D:\Documents\modeling\Bengin\Bengin_set1_3D_arrays"
ren = os.listdir(path)
length = len(ren)
for m in range(7, length):
    print(ren[m])
    print(m)
    addition1 = os.path.join(path,ren[m])
    LMLO = 600
    CC = 600
    FTF = 600
    graphs = graph_array(addition1)
        
    output1 = r"D:\Documents\modeling\Bengin\3D_graphs\%s_1.png"%(ren[m].replace('.npy',''))
    output2 = r"D:\Documents\modeling\Bengin\3D_graphs\%s_2.png"%(ren[m].replace('.npy',''))
    
                



    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(graphs[:,0],graphs[:,1],graphs[:,2])
    ax.set_xlim3d(FTF,0)
    ax.set_ylim3d(0,CC)
    ax.set_zlim3d(0, LMLO)
    ax.view_init(90,80)
    fig1.savefig(output1)
    plt.close(fig1)
    print(m)
    fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(graphs[:,0],graphs[:,1],graphs[:,2])
    ax.set_xlim3d(FTF,0)
    ax.set_ylim3d(0,CC)
    ax.set_zlim3d(0, LMLO)
    ax.view_init(50,40)
    fig2.savefig(output2)
    plt.close(fig2)
    print(m)