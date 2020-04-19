
import numpy as np
import matplotlib.pyplot as plt


LMLO = 712
CC = 704

multidimensional_array_top_left = np.load('3D_prob_walker_array.npy')
multidimensional_array_centre = np.load('3D_prob_walker_array1.npy')

Total_array = np.zeros((712,700,704))

for jj in range(0,704):
    for ii in range(0,700):
        for kk in range(0,712):
            Total_array[kk,ii,jj] = multidimensional_array_centre[kk,ii,jj] + multidimensional_array_top_left[kk,ii,jj]



np.save('3D_prob_total_array',Total_array)

LMLO_array = np.zeros((LMLO, 700))
CC_array = np.zeros((CC, 700))
for kk in range(0, LMLO):
    for ii in range(0, 700):
        val = np.sum(Total_array[kk,ii,0:704])
        LMLO_array[kk,ii] = val
        
for jj in range(0, CC):
    for ii in range(0, 700):
        val = np.sum(Total_array[0:712,ii,jj])
        CC_array[jj,ii] = val
        
plt.imshow(LMLO_array)
plt.colorbar()