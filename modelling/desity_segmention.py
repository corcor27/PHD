import numpy as np
import pydicom as dicom

import matplotlib.pyplot as plt

UB = 5000
LB = 0
gradient_Threashold = 0.5
path = r"C:\Users\cory1\Documents\test-folder\data_set\dicom_M\0146-2.dcm"

def image_position(image,x1,x2,z1,z2, LB):
    Beginning_image = dicom.dcmread(path)
    beginning_image = Beginning_image.pixel_array
    
    x1ml = x1-50
    x2ml = x2+50
    z1ml = z1-50
    z2ml = z2+50
    diffxml = x2ml - x1ml
    diffzml = z2ml - z1ml
    print(diffxml,diffzml)
    
    array = np.zeros((diffzml, diffxml))
    for j in range(z1ml, z2ml):
        for k in range(x1ml, x2ml):
            array[j-z1ml,k-x1ml] = beginning_image[j,k]
    density = np.zeros((diffzml,diffxml))
    u = 0
    for i in range(u, diffzml-u):
        for j in range(u, diffxml-u):
            lum = array[i,j]
            if LB <= lum:
                density[i,j] = lum
            else:
                density[i,j] = 0
    return diffzml, diffxml, density, array

def values_array(array, diffzml, diffxml,rr):
    list = []
    for ii in range(0,rr):
        x = np.random.randint(0,high = diffzml)
        z = np.random.randint(0,high = diffxml)
        lum = array[x,z]
        list.append(lum)
    gg = sum(list)/ rr
    return gg
    
def values2_array(array, diffzml, diffxml,rr):
    count = 0
    list = []
    while count < 1:
        
        x = np.random.randint(0,high = diffzml)
        z = np.random.randint(0,high = diffxml)
        lum = array[x,z]
        if lum >= 1:
            list.append(lum)
        if len(list) == rr:
            count += 1
    gg = sum(list)/ rr
    return gg

def density_value(array, diffzml, diffxml):
    area = round(((diffzml*diffxml)/100)*2)
    list = []
    for rr in range(100,area):
        ff = values_array(array, diffzml, diffxml,rr)
        list.append(ff)
    gg = sum(list)/len(list)
    return gg

def density2_value(array, diffzml, diffxml):
    area = 0
    list = []
    for i in range(0, diffzml):
        for j in range(0, diffxml):
            if array[i,j] >= 1:
                area += 1 
    ratio = round((area/100)*2)
    for rr in range(10,ratio):
        ff = values2_array(array, diffzml, diffxml,rr)
        list.append(ff)
    gg = sum(list)/len(list)
    return gg

def line_graph(array, diffzml, diffxml):
    list1 = []
    for kk in range(0, diffxml):
        list2 = []
        for ii in range(0,diffzml):
            lum = array[ii,kk]
            list2.append(lum)
        lum1 = sum(list2)
        list1.append(lum1)
        
    return list1

def patch(array, ii, kk):
    list = []
    for zz in range(-50,50):
        for dd in range(-50,50):
            x = ii + zz
            z = kk + dd
        lum = array[x,z]
        list.append(lum)
    val = sum(list)
    if val != 0:
        value = val/len(list)
        return value
    else:
        return 0
        
    
    
def max_value(array, diffzml, diffxml):
    new_array = np.zeros((diffzml,diffxml))
    for kk in range(50, diffxml-50):
        for ii in range(50, diffzml-50):
            val = patch(array, ii, kk)
            new_array[ii,kk] = val
    max_val = np.amax(new_array)
    
    for kk in range(0, diffxml):
        for ii in range(0, diffzml):
            if new_array[ii,kk] == max_val:
                x = ii
                z = kk
                print(x,z)
    
    return x , z
    
def centre_image(image,x1,x2,z1,z2, posx, posz, LB):
    new_diffzml = 600
    new_diffxml = 600
    print(posx, posz)
    array = np.zeros((new_diffzml,new_diffxml))
    Beginning_image = dicom.dcmread(image)
    beginning_image = Beginning_image.pixel_array
    halfx = int(new_diffxml/2)
    halfz = int(new_diffzml/2)
    centrex = int(x1) + posz - 50
    centrez = int(z1) + posx - 50
    
    new_x1 = centrex - halfx
    new_x2 = centrex + halfx
    new_z1 = centrez - halfz
    new_z2 = centrez + halfz
   
    for j in range(new_z1, new_z2):
        for k in range(new_x1, new_x2):
            array[j-new_z1,k-new_x1] = beginning_image[j,k]
    density = np.zeros((new_diffzml,new_diffxml))
    u = 0
    for i in range(u, new_diffzml-u):
        for j in range(u, new_diffxml-u):
            lum = array[i,j]
            if LB <= lum:
                density[i,j] = lum
            else:
                density[i,j] = 0
    return new_diffzml, new_diffxml, density, array        

x1 = 283
x2 = 691
z1 = 1699
z2 = 2011

dcm_sample = image_position(path,x1,x2,z1,z2, LB)
density = density_value(dcm_sample[2], dcm_sample[0], dcm_sample[1])
new_sample = image_position(path,x1,x2,z1,z2, density)

density2 = density2_value(new_sample[2], new_sample[0], new_sample[1])
another_sample = image_position(path,x1,x2,z1,z2, density2)
#density3 = density2_value(another_sample[2], new_sample[0], new_sample[1])
#last_sample = image_position(path,x1,x2,z1,z2, density3)
#create_line = line_graph(another_sample[2], another_sample[0], another_sample[1] )
max_line = max_value(new_sample[2], new_sample[0], new_sample[1])
new_sample_low = centre_image(path,x1,x2,z1,z2,max_line[0],max_line[1], LB)
#max_low_line = max_value(new_sample_low[2], new_sample_low[0],new_sample_low[1])
#print(max_low_line[0], max_low_line[1])
plt.imshow(new_sample_low[2])
#plt.imshow(last_sample[2])
#plt.imshow(dcm_sample[2])
#plt.colorbar()

#plt.plot(create_line)
#plt.plot(max_line)