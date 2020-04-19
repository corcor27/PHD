import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import os

def LMLO1(file):
    multidimensional_array = np.load(file)
    LMLO_array = np.zeros((600, 600))
    for kk in range(1, 600-1):
        for ii in range(1, 600-1):
            val = np.sum(multidimensional_array[kk,ii,0:600])
            if val > 0:
                LMLO_array[kk,ii] = val
                
    del multidimensional_array
    return LMLO_array

def CC1(file):
    multidimensional_array = np.load(file)
    CC_array = np.zeros((600, 600))
    for jj in range(1, 600):
        for ii in range(1, 600):
            val = np.sum(multidimensional_array[0:600,ii,jj])
            if val > 0:
                CC_array[jj,ii] = val
    del multidimensional_array
    return CC_array

def FTF1(file):
    multidimensional_array = np.load(file)
    FTF_array = np.zeros((600, 600))
    for kk in range(0, 600):
        for jj in range(0, 600):
            val = np.sum(multidimensional_array[kk,0:600,jj])
            if val > 0:
                FTF_array[kk,jj] = val
    del multidimensional_array
    return FTF_array
    


path = r"D:\Documents\modeling\Bengin\threashold_testing_with_area"
ren = os.listdir(path)
length = len(ren)
for m in range(0, length):
    addition1 = os.path.join(path,ren[m])
    print(ren[m])
    LMLO = LMLO1(addition1)
    CC = CC1(addition1)
    FTF = FTF1(addition1)
    output1 = r"D:\Documents\modeling\Bengin\area_threashold_test_breakdown\%s_LMLO.png"%(ren[m].replace('.npy',''))
    output2 = r"D:\Documents\modeling\Bengin\area_threashold_test_breakdown\%s_CC.png"%(ren[m].replace('.npy',''))
    output3 = r"D:\Documents\modeling\Bengin\area_threashold_test_breakdown\%s_FTF.png"%(ren[m].replace('.npy',''))
    plt.imsave(output1, LMLO)
    plt.imsave(output2, CC)
    plt.imsave(output3, FTF)
    del LMLO
    del CC
    del FTF
    
    
    
    




      




