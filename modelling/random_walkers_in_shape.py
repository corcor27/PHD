import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def prob_value(array, ii, jj, kk, t):
    lum = array[kk,ii,jj]
    p = int(round(100*lum/t))
    #p = 100
    return p
def lum_value(array, ii, jj, kk):
    lum = array[kk,ii,jj]
    return lum

def start_positions(array1, array2,  No_walkers, LB):
    dimesional_array = np.load(array1)
    dimesional_array2 = np.load(array2)
    LMLO = dimesional_array.shape[0]
    CC = dimesional_array.shape[1]
    FTF = dimesional_array.shape[2]
    positions = np.zeros((No_walkers, 3),dtype=int)
    count = 0
    while count < No_walkers:
        x = np.random.randint(0,high = FTF,size = 1)
        y = np.random.randint(0,high = CC,size = 1)
        z = np.random.randint(0,high = LMLO,size = 1)
        if dimesional_array[z,x,y] >= LB and dimesional_array2[z,x,y] >= LB:
            positions[count, :] = z, x, y
            count += 1 
    
    return positions

def position_check(ii, jj, kk, LMLO, CC, FTF):
    if 1 <= ii <= FTF-1 and 1 <= jj <= CC-1 and 1 <= kk <= LMLO-1:
        return 1
    else:
        return 0
    
def create_walkers(array,No_walkers,steps, starts_array, bound, LB):
    dimesional_array = np.load(array)
    LMLO = dimesional_array.shape[0]
    FTF = dimesional_array.shape[1]
    CC = dimesional_array.shape[2]
    walker_array = np.zeros((LMLO,FTF,CC))
    t = np.amax(dimesional_array)
    for ww in range(0, No_walkers):
        print(ww)
        xlist = [starts_array[ww, 1]]
        ylist = [starts_array[ww, 2]]
        zlist = [starts_array[ww, 0]]
        walker_array[zlist[-1],xlist[-1],ylist[-1]] = 1
        for rr in range(0, steps):
            x_val = np.random.randint(-1,high = 2,size = 1)
            y_val = np.random.randint(-1,high = 2,size = 1)
            z_val = np.random.randint(-1,high = 2,size = 1)
            ii = xlist[-1] + x_val[0]
            jj = ylist[-1] + y_val[0]
            kk = zlist[-1] + z_val[0]
            pp = position_check(ii, jj, kk, LMLO, CC, FTF)
            if pp == 1:
                prob_val = prob_value(dimesional_array, ii, jj, kk, t)
                lum = lum_value(dimesional_array, ii, jj, kk)
                s = np.random.randint(-1,high = 101,size = 1)
                if rr <= bound:
                    if s - prob_val <= 0 and lum >=LB:
                        xlist.append(ii)
                        ylist.append(jj)
                        zlist.append(kk)
                        walker_array[kk,ii,jj] += 1
                else: 
                    if s - prob_val <= 0:
                        xlist.append(ii)
                        ylist.append(jj)
                        zlist.append(kk)
                        walker_array[kk,ii,jj] += 1
    return walker_array


LB = 1001
No_walkers = 100
steps = 20000
bound = 20000
path = r"D:\Documents\modeling\Bengin\Bengin_set1_3D_arrays"
ren = os.listdir(path)
length = len(ren)
for m in range(8, length,2):
    v = m+1
    print(ren[m])
    print(ren[v])
    addition1 = os.path.join(path,ren[m])
    addition2 = os.path.join(path,ren[v])
    start_positions = start_positions(addition2,addition1, No_walkers, LB)
    create_walker_array = create_walkers(addition1, No_walkers,steps, start_positions, bound, LB)
    val = np.sum(create_walker_array)
    print(val)
    output = r"D:\Documents\modeling\Bengin\walker_arrays_threashold1001\%s_%d_%d_%d" %(ren[m].replace('.npy',''), No_walkers, steps, LB)
    np.save(output,create_walker_array)
    
        
    









        





