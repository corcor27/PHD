import numpy as np
import cv2
import pydicom as dicom
from skimage import exposure
import matplotlib.pyplot as plt
import os
import csv
from scipy import ndimage

LB = 2800 #1800

path1 = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\0145-1.dcm"
path2 = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\0145-2.dcm"
#path = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\1509-2.dcm"
#ren = os.listdir(path)
#length = len(ren)
#for m in range(0,length):
    #addition = os.path.join(path,ren[m])

def image_position(path,x1,x2,z1,z2, LB):
    Beginning_image = dicom.dcmread(path)
    beginning_image = Beginning_image.pixel_array
    
    x1ml = x1 
    x2ml = x2 
    z1ml = z1 
    z2ml = z2 
    diffxml = x2ml - x1ml
    diffzml = z2ml - z1ml
    print(diffxml,diffzml)
    array = np.zeros((diffzml, diffxml))
    for j in range(z1ml, z2ml):
        for k in range(x1ml, x2ml):
            array[j-z1ml,k-x1ml] = beginning_image[j,k]
    density = np.zeros((diffzml,diffxml))
    u = 0
    UB = np.amax(array)
    for i in range(u, diffzml-u):
        for j in range(u, diffxml-u):
            lum = array[i,j]
            if LB <= lum <= UB:
                density[i,j] = lum
            else:
                density[i,j] = 0
    
    pyrlvl1 = np.zeros((diffzml, diffxml))
    pyrlvl1 = ndimage.filters.gaussian_filter(density, 0.5)
    #new_image = density + pyrlvl2
    
    return diffzml, diffxml, pyrlvl1, density

    


def fallers(array, diffzml, diffxml, x0, z0, number,LB):
    space = np.zeros((diffzml, diffxml))
    pos_x = []
    pos_z = []
    pos_x.append(x0)
    pos_z.append(z0)
    space[x0,z0] = 1
    for rr in range(0,number):
        x = pos_x[rr]
        z = pos_z[rr]
        di = np.random.randint(-1,high = 2)
        dk = np.random.randint(-1,high = 2)
        new_x = int(x + di)
        new_z = int(z + dk)
        
        if new_x <= diffzml-1 and new_z <= diffxml-1:
            old_pos = round(array[x,z])
            new_pos = round(array[new_x,new_z])
            E = round((old_pos/100)*5)
        
            if new_pos >= old_pos - E and new_pos >= LB:
                space[new_x,new_z] = 1
                pos_x.append(new_x)
                pos_z.append(new_z)
            else:
                break
        else:
            break
    
    return space

            
def find_max(array, diffzml, diffxml, tt) :
    for ii in range(0,diffzml):
        for kk in range(0,diffxml):
            if array[ii,kk] == tt:
                x = int(ii)
                z = int(kk)
    return x,z
    
def map_space(array, diffzml, diffxml, number, LB):
    tt = np.max(array)
    space = np.zeros((diffzml, diffxml))
    x0 = find_max(array, diffzml, diffxml, tt)[0]
    z0 = find_max(array, diffzml, diffxml, tt)[1]
    for ll in range(0,10000):        
        fall = fallers(array, diffzml, diffxml, x0, z0, number, LB)
        space = np.add(space, fall)
                
    new_space = np.zeros((diffzml, diffxml))
    for ii in range(0,diffzml):
        for kk in range(0,diffxml):
            if space[ii,kk] >= 1:
                new_space[ii,kk] = array[ii,kk]
            
    
    return new_space

def lum_curve(array, diffzml, diffxml):
    new_array = []
    x = np.linspace(0,diffxml,diffxml)
    for kk in range(0,diffxml):
        val = np.amax(array[:,kk])
        new_array. append(val)
    return new_array, x
    
dcm_sample1 = image_position(path1, 1818,1949,1436,1565, LB)
dcm_sample2 = image_position(path2, 1372,1551,1492,1674, LB)
#dcm_sample = image_position(path, 1549,1806,336, 610, LB)


#dcm_sample = image_position(path, 1033,1217,1782,1966, LB)
#dcm_sample = image_position(path, 1102,1281,2864,3040)

#im_LMLO = map_space(dcm_sample2[2], dcm_sample2[0], dcm_sample2[1], 1000, LB)
image1 = lum_curve(dcm_sample1[2], dcm_sample1[0], dcm_sample1[1])
image2 = lum_curve(dcm_sample2[2], dcm_sample2[0], dcm_sample2[1])
plt.plot(image1[1],image1[0])
plt.plot(image2[1],image2[0])
#plt.imshow(dcm_sample2[2], cmap='gray')

#plt.hist(dcm_sample1[3])
#plt.imshow(dcm_sample[2])
#plt.colorbar()
#plt.imshow(dcm_sample[2], cmap='gray')
#plt.imshow(LMLO,cmap='gray')



        

   
    
    
    
