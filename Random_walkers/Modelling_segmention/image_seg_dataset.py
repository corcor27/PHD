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

File = '/home/cot12/Documents/test-folder/data_set/Malignant/Dataset3.csv'
path = '/home/cot12/Documents/test-folder/data_set/dicom_M'

def image_position(path,x1,x2,z1,z2, LB):
    Beginning_image = dicom.dcmread(path)
    beginning_image = Beginning_image.pixel_array
    
    x1ml = x1 
    x2ml = x2 
    z1ml = z1
    z2ml = z2
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
    return diffzml, diffxml, density

def values_array(array, diffzml, diffxml,rr):
    list = []
    for ii in range(0,rr):
        print(rr)
        x = np.random.randint(0,high = diffzml)
        z = np.random.randint(0,high = diffxml)
        lum = array[x,z]
        list.append(lum)
    gg = sum(list)/ rr
    return gg
    


def density_value(array, diffzml, diffxml):
    area = round(((diffzml*diffxml)/100)*2)
    list = []
    for rr in range(100,area):
        print(rr)
        ff = values_array(array, diffzml, diffxml,rr)
        list.append(ff)
    gg = max(list)
    return gg
    
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


ren = os.listdir(path)
length = len(ren)
print(length)
for m in range(0, length):
    addition = os.path.join(path,ren[m])
    print(ren[m])
    values = csv_file_read(m)
    print(values)
    dcm_sample = image_position(addition,values[0],values[1],values[2],values[3], LB)
    density = density_value(dcm_sample[2], dcm_sample[0], dcm_sample[1])
    new_sample = image_position(dcm_sample[2], dcm_sample[0], dcm_sample[1], density)
    density2 = density2_value(new_sample[2], values[0],values[1],values[2],values[3])
    another_sample = image_position(addition,values[0],values[1],values[2],values[3], density2)
    outfile = '/home/cot12/Documents/test-folder/data_set/Malignant/dropbox/%s' %(ren[m])
    cv2.imwrite(outfile, new_sample[2])
    
    



