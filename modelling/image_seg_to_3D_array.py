import numpy as np
import pydicom as dicom
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
import csv
import os

UB = 5000
LB = 0
gradient_Threashold = 0.5
input_size = 600

File = r"C:\Users\cory1\Documents\test-folder\data_set\Bengin\Bengin_set.csv"
path = r"C:\Users\cory1\Documents\test-folder\data_set\Bengin\dicom_M"

def image_position(image,x1,x2,z1,z2, LB):
    Beginning_image = dicom.dcmread(image)
    beginning_image = Beginning_image.pixel_array
    
    x1ml = int(x1)-50
    x2ml = int(x2)+50
    z1ml = int(z1)-50
    z2ml = int(z2)+50
    diffxml = x2ml - x1ml
    diffzml = z2ml - z1ml
    
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




def density_value(array, diffzml, diffxml):
    area = diffzml * diffxml
    total_value = np.sum(array)
    ss = total_value/ area
    print(ss)
    return ss

def density2_value(array, diffzml, diffxml):
    area = 0
    list = []
    for i in range(0, diffzml):
        for j in range(0, diffxml):
            if array[i,j] >= 1:
                list. append(array[i,j])
                area += 1
    gg = (sum(list)/area)
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
                
    return x , z
           
def csv_file_read(m):
    with open(File) as csvfile:
        readcsv = csv.reader(csvfile, delimiter = ',') #read file 
        pos_x1 = []
        pos_x2 = []
        pos_z1 = []
        pos_z2 = []
        for row in readcsv:
            x1 = row[12]
            x2 = row[13]
            z1 = row[14]
            z2 = row[15]
            pos_x1.append(x1)
            pos_x2.append(x2)
            pos_z1.append(z1)
            pos_z2.append(z2)
    
        pos_x1 [0] = '0'# ignore first elememt of list
        pos_x2 [0] = '0'
        pos_z1 [0] = '0'
        pos_z2 [0] = '0'
        pos_x1.pop(0)
        pos_x2.pop(0)
        pos_z1.pop(0)
        pos_z2.pop(0)
        val1 = pos_x1[m]
        val2 = pos_x2[m]
        val3 = pos_z1[m]
        val4 = pos_z2[m]
    return val1,val2,val3,val4

def centre_image(image,x1,x2,z1,z2, posx, posz, LB, array_size, value):
    new_diffzml = array_size
    new_diffxml = array_size
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
    if value == 1:
        new_array = np.rot90(density,  k=3, axes=(0, 1))
    else:
        new_array = density
    return new_diffzml, new_diffxml, new_array, array  

def ave_pos(p1,p2,p3,p4):
    val = round((p1+p2+p3+p4)/4)
    return val

def create3D_array(array1, array2, array_size):
    dimesional_array = np.zeros((array_size,array_size,array_size))
    list = []
    for k in range(1,array_size-1):
        for j in range(1, array_size-1):
            for i in range(1,array_size-1):
                if array2[j,i] > 10 and array1[k,i] > 10:
                    val = (array1[j,i]+array2[k,i])/2
                    dimesional_array[k,i,j] = val
                    list.append(val)
                    
    lowest = min(list)
    print(lowest)         
    for k in range(1,input_size-1):
        for j in range(1, input_size-1):
            for i in range(1,input_size-1):
                if dimesional_array[k,i,j] == 0:
                    dimesional_array[k,i,j] = 1000
    return dimesional_array, lowest

ren = os.listdir(path)
length = len(ren)
for m in range(0, length,2):
    v = m+1
    print(ren[m])
    addition1 = os.path.join(path,ren[m])
    values1 = csv_file_read(m)
    base_sample1 = image_position(addition1,values1[0],values1[1],values1[2],values1[3], LB)
    base_density1 = density_value(base_sample1[2], base_sample1[0], base_sample1[1])
    lvl1_sample1 = image_position(addition1,values1[0],values1[1],values1[2],values1[3], base_density1)
    lvl1_density1 = density2_value(lvl1_sample1[2], lvl1_sample1[0], lvl1_sample1[1])
    lvl2_sample1 = image_position(addition1,values1[0],values1[1],values1[2],values1[3], lvl1_density1)
    lvl2_density1 = density2_value(lvl2_sample1[2], lvl2_sample1[0],lvl2_sample1[1])
    lvl3_sample1 = image_position(addition1,values1[0],values1[1],values1[2],values1[3], lvl2_density1)
    base_sample_1 = max_value(base_sample1[2], base_sample1[0],base_sample1[1])
    lvl1_sample_1 = max_value(lvl1_sample1[2], lvl1_sample1[0], lvl1_sample1[1])
    lvl2_sample_1 = max_value(lvl2_sample1[2], lvl2_sample1[0], lvl2_sample1[1])
    lvl3_sample_1 = max_value(lvl3_sample1[2], lvl3_sample1[0], lvl3_sample1[1])
    positionx1 = ave_pos(base_sample_1[0],lvl1_sample_1[0],lvl2_sample_1[0],lvl3_sample_1[0])
    positionz1 = ave_pos(base_sample_1[1],lvl1_sample_1[1],lvl2_sample_1[1],lvl3_sample_1[1])
    #imaged_centred1 = centre_image(addition1,values1[0],values1[1],values1[2],values1[3],positionx1,positionz1, base_density1, input_size, 0)
    imaged_centred1 = centre_image(addition1,values1[0],values1[1],values1[2],values1[3],positionx1,positionz1, lvl1_density1, input_size, 0)
    
    print(ren[v])
    addition2 = os.path.join(path,ren[v])
    values2 = csv_file_read(v)
    base_sample2 = image_position(addition2,values2[0],values2[1],values2[2],values2[3], LB)
    base_density2 = density_value(base_sample2[2], base_sample2[0], base_sample2[1])
    lvl1_sample2 = image_position(addition1,values2[0],values2[1],values2[2],values2[3], base_density2)
    lvl1_density2 = density2_value(lvl1_sample2[2], lvl1_sample2[0], lvl1_sample2[1])
    lvl2_sample2 = image_position(addition2,values2[0],values2[1],values2[2],values2[3], lvl1_density2)
    lvl2_density2 = density2_value(lvl2_sample2[2], lvl2_sample2[0],lvl2_sample2[1])
    lvl3_sample2 = image_position(addition2,values2[0],values2[1],values2[2],values2[3], lvl2_density2)
    base_sample_2 = max_value(base_sample2[2], base_sample2[0],base_sample2[1])
    lvl1_sample_2 = max_value(lvl1_sample2[2], lvl1_sample2[0], lvl1_sample2[1])
    lvl2_sample_2 = max_value(lvl2_sample2[2], lvl2_sample2[0], lvl2_sample2[1])
    lvl3_sample_2 = max_value(lvl3_sample2[2], lvl3_sample2[0], lvl3_sample2[1])
    positionx2 = ave_pos(base_sample_2[0],lvl1_sample_2[0],lvl2_sample_2[0],lvl3_sample_2[0])
    positionz2 = ave_pos(base_sample_2[1],lvl1_sample_2[1],lvl2_sample_2[1],lvl3_sample_2[1])
    #imaged_centred2 = centre_image(addition1,values2[0],values2[1],values2[2],values2[3],positionx2,positionz2,base_density2, input_size, 0)
    imaged_centred2 = centre_image(addition1,values2[0],values2[1],values2[2],values2[3],positionx2,positionz2,lvl1_density2, input_size, 0)
   
    
    create_array = create3D_array(imaged_centred1[2], imaged_centred2[2], input_size)
    output = r"D:\Documents\modeling\Bengin\Bengin_set1_3D_arrays\%s_2_lvl1" %(ren[m].replace('.dcm',''))
    np.save(output,create_array[0])  
    



