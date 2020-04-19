import cv2
import numpy as np
import matplotlib.pyplot as plt


UB = 255
LB = 150
input_size = 800




def image_position(image,x1,x2,z1,z2,UB,LB):
    Beginning_image = cv2.imread(image,0)
    beginning_image = Beginning_image
    
    x1ml = x1 - 50
    x2ml = x2 + 50
    z1ml = z1 - 50
    z2ml = z2 + 50
    diffxml = x2ml - x1ml
    diffzml = z2ml - z1ml
    print(diffxml,diffzml)
    array = np.zeros((diffzml, diffxml))
    for j in range(z1ml, z2ml):
        for k in range(x1ml, x2ml):
            array[j-z1ml,k-x1ml] = beginning_image[j,k]
    density = np.zeros((diffzml,diffxml))
    u = 5
    for i in range(u, diffzml-u):
        for j in range(u, diffxml-u):
            lum = np.sum(array[i-u:i+u,j-u:j+u])/(len(array[i-u:i+u,j-u:j+u])**2)
            if LB <= lum <= UB:
                density[i,j] = lum
            else:
                density[i,j] = 0
    
    return diffzml, diffxml, density

def intial(image,x1,x2,z1,z2,UB,LB):
    transfer = image_position(image,x1,x2,z1,z2,UB,LB)
    diffzml = transfer[0]
    diffxml = transfer[1]
    return transfer[2], diffzml, diffxml

image1 = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\dicom_convert\0145-1.png"
image2 = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\dicom_convert\0145-2.png"

LMLO = intial(image1,1818,1949,1436,1565,UB,LB)
#CC = intial(image2,1372,1551,1492,1674,UB,LB)

def dis_check_LMLO(array, diffzml, diffxml, input_size):
    New_image = np.zeros((input_size, input_size))
    t = np.amax(array)
    listx = []
    for kk in range(0, diffzml):
        for ii in range(0, diffxml):
            if array[kk,ii] == t:
                x0 = int(ii)
                print(x0)
                listx.append(x0)
    half = round(input_size/2)
    ss = round(half - listx[0])
    if ss > 0:
        for kk in range(0, diffzml):
            for ii in range(0,diffxml):
                New_image[kk,ii-ss] = array[kk,ii]
    else:
        for kk in range(0, diffzml):
            for ii in range(0,diffxml):
                New_image[kk,ii+ss] = array[kk,ii]
    return New_image


im_LMLO = dis_check_LMLO(LMLO[0], LMLO[1], LMLO[2], input_size)

#im_CC = dis_check_CC(CC[0], CC[1], CC[2], input_size)

plt.imshow(im_LMLO[0])

"""
dimesional_array = np.zeros((input_size,input_size, input_size))


for k in range(1,input_size-1):
    for j in range(1, input_size-1):
        for i in range(1,input_size-1):
            if im_CC[j,i] > LB and im_LMLO[k,i] > LB:
                dimesional_array[k,i,j] = (im_CC[j,i]+im_LMLO[k,i])/2
            else:
                dimesional_array[k,i,j]= LB

for k in range(1,input_size-1):
    for j in range(1, input_size-1):
        for i in range(1,input_size-1):
            if dimesional_array[k,i,j] == 0:
                dimesional_array[k,i,j] = LB
                
           

np.save('3D_array_test1_diff_images',dimesional_array)

"""


