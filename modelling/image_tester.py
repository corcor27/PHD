import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv




UB = 255

LB = 150

input_size = 800



def image_position(image,x1,x2,z1,z2,UB,LB):
    Beginning_image = image
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
    return transfer[2]

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


File = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\Dataset3.csv"
path = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\dicom_convert"

ren = os.listdir(path)
length = len(ren)
for m in range(0,length):
    addition = os.path.join(path,ren[m])
    image = cv2.imread(addition,0)
    Beginning_image = image
    print(ren[m])
    l = np.amax(Beginning_image)
    print(l)
    values = csv_file_read(m)
    print(values)
    LMLO = intial(Beginning_image,int(values[0]),int(values[1]),int(values[2]),int(values[3]),UB,LB)    
    outfile = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\output\%s" % (ren[m])
    #plt.imshow(LMLO)
    cv2.imwrite(outfile, LMLO)
    
    
    
    