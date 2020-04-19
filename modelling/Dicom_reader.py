import numpy as np
import cv2
import pydicom as dicom
from scipy import ndimage as ndi
from skimage import exposure
import matplotlib.pyplot as plt
import os
import csv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
UB = 1200
LB = 500
path = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\1556-1.dcm"
#path = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\1509-2.dcm"
#ren = os.listdir(path)
#length = len(ren)
#for m in range(0,length):
    #addition = os.path.join(path,ren[m])

def image_position(image,x1,x2,z1,z2, LB):
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
    u = 2
    UB = np.amax(array)
    LB = LB
    for i in range(u, diffzml-u):
        for j in range(u, diffxml-u):
            lum = array[i,j]
            if LB <= lum <= UB:
                density[i,j] = lum
            else:
                density[i,j] = 0
                
    pyrlvlx = np.zeros((diffzml,diffxml))
    pyrlvly = np.zeros((diffzml,diffxml))
    Grmaglvl = np.zeros((diffzml,diffxml))
    pyrlvlx[:,:] = ndi.sobel(density, 0)
    pyrlvly[:,:] = ndi.sobel(density, 1)
    Grmaglvl = (((pyrlvlx)**2)+((pyrlvly)**2))**0.5
    
    return diffzml, diffxml, Grmaglvl, density   

  
def fallers(array, diffzml, diffxml, x0, z0, number):
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
        old_pos = round(array[x,z])
        new_pos = round(array[new_x,new_z])
        
        if new_pos <= old_pos:
            space[new_x,new_z] = 1
            pos_x.append(new_x)
            pos_z.append(new_z)
        else:
            break
    return space

            
def find_max(array, diffzml, diffxml, tt) :
    #for ii in range(0,diffzml):
        #for kk in range(0,diffxml):
            #if array[ii,kk] == tt:
    x0 = np.random.randint(-10,high = diffxml - 10)
    z0 = np.random.randint(-10,high = diffzml - 10)
    return x0,z0
    
def map_space(array, diffzml, diffxml, number, LB):
    tt = np.amax(array)
    space = np.zeros((diffzml, diffxml))
    x0 = find_max(array, diffzml, diffxml, tt)[0]
    z0 = find_max(array, diffzml, diffxml, tt)[1]
    for ll in range(0,100):        
        fall = fallers(array, diffzml, diffxml, x0, z0, number)
        space = np.add(space, fall)
                
    new_space = np.zeros((diffzml, diffxml))
    for ii in range(0,diffzml):
        for kk in range(0,diffxml):
            if space[ii,kk] >= 1:
                new_space[ii,kk] = array[ii,kk]
            
    
    return new_space

def calulate_gradient(array, kk, ii, max_value, gradient_Threashold):
    dx = (array[ii,kk+10] - array[ii,kk-10]) * 0.5 / max_value
    dy = (array[ii+10,kk] - array[ii+10,kk]) * 0.5 / max_value
    dxy = (array[ii+5,kk+5] - array[ii-5,kk-5]) * 0.5 / max_value
    dyx = (array[ii-5,kk+5] - array[ii+5,kk-5]) * 0.5 / max_value
    x = np.sum(dx + dy + dxy + dyx)
    val = 4*gradient_Threashold
    if x >= -val:
        return 1
    else:
        return 0
 
def check(array, kk, ii, Threashold):
    uu = array[ii,kk]
    if uu >= Threashold:
        return 1
    else:
        return 0

def check_again(array, kk, ii, UPPER_BOUND):
    uu = array[ii,kk]
    if uu >= UPPER_BOUND:
        return 1
    else:
        return 0
    
def region_definining(array, diffzml, diffxml, LB, UB):
    max_value = np.amax(array)
    space = np.zeros((diffzml, diffxml))
    Threashold = LB
    UPPER_BOUND = UB
    gradient_Threashold= 0.05
    samples = []
    count = 0
    for ii in range(10,diffzml-10):
        for kk in range(10,diffxml-10):
            gg = calulate_gradient(array, kk, ii, max_value, gradient_Threashold)
            ll = check(array, kk, ii, Threashold)
            dd = check_again(array, kk, ii, UPPER_BOUND)
            if gg == 1 and ll == 1 and dd == 1:
                space[ii,kk] = array[ii,kk]
                count += 1
    print(count)           
    return space
                
    

dcm_sample = image_position(path, 1642,2272, 1119, 1498, LB)
#dcm_sample = image_position(path, 1033,1217,1782,1966, LB)
#dcm_sample = image_position(path, 1102,1281,2864,3040)
LMLO = region_definining(dcm_sample[2], dcm_sample[0], dcm_sample[1], LB,UB)

im_LMLO = map_space(LMLO, dcm_sample[0], dcm_sample[1], 1000, LB)

#plt.imshow(LMLO, cmap='gray')
#plt.imshow(dcm_sample[2])
plt.imshow(im_LMLO)



        

   
    
    
    
