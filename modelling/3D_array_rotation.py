import numpy as np
import pydicom as dicom
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
import csv
import os


def graph_array(addition1, kk):
    dimesional_array = np.load(addition1)
    
    LMLO = dimesional_array.shape[0]
    CC = dimesional_array.shape[1]
    FTF = dimesional_array.shape[2]
    rotated_array = np.zeros((LMLO ,CC,FTF))
    for i in range(0, LMLO):
        transfer_array = dimesional_array[:,i,:]
        new = np.rot90(transfer_array,  k=kk, axes=(0, 1))
        rotated_array[:,i,:] = new
    return rotated_array
    
        
def angle(kk):
    if kk == 1:
        return "90"
    elif kk == 2:
        return "180"
    elif kk == 3:
        return "270"



path = r"D:\Documents\modeling\3D_rotations"
ren = os.listdir(path)
length = len(ren)
for m in range(14, length):
    print(ren[m])
    print(m)
    addition1 = os.path.join(path,ren[m])
    for kk in range(3, 4):
        graphs = graph_array(addition1, kk)
        angle = angle(kk)
        print(angle)
        output = r"D:\Documents\modeling\3D_rotations\1579\%s_%s_degrees" %(ren[m].replace('.npy',''), angle)
        np.save(output,graphs)