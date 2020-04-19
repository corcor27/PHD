import cv2
import numpy as np
import matplotlib.pyplot as plt


UB = 255
LB = 150
input_size = 800



def positions(i,k,array,clusterA):
# functions creates array of new positions
    new_x = []
    new_z = []
    for di in range(-1,2):
        for dk in range(-1,2):
            x = int(i+di)
            z = int(k+dk)
            val_new = position_check_A(x,z,clusterA)
            val_old = position_check_ini(x,z,array)
            
            if val_old == 1 and val_new == 0:
                new_x.append(x)
                new_z.append(z)
    return new_x, new_z

def position_check_A(x,z,clusterA):
    if clusterA[x,z] > 0:
        return 1
    else:
        return 0
    
def position_check_ini(x,z, array):
    if array[x,z] > 0:
        return 1
    else:
        return 0
    
def Random_position(array, clusterA, diffzml,diffxml):
    x_p = []
    z_p = []
    while len(x_p) == 0:
        x = np.random.randint(1,high = diffzml-2)
        z = np.random.randint(1,high = diffzml-2)
        val_new = position_check_A(x,z,clusterA)
        val_old = position_check_ini(x,z,array)
        if val_old == 1 and val_new == 0:
            x_p.append(x)
            z_p.append(z)
    return x_p[0], z_p[0]


def cluster_1(array,diffzml,diffxml, clusterA, pp):
    x_list = []
    z_list = []
    initial_pos = Random_position(array, clusterA, diffzml,diffxml)
    initial_x = initial_pos[0]
    initial_z = initial_pos[1]
    x_list.append(initial_x)
    z_list.append(initial_z)
    clusterA[initial_x,initial_z] = pp
    
    while len(x_list) > 0:
        i1 = x_list[0]
        k1 = z_list[0]
        i_k_pos = positions(i1,k1,array,clusterA)
        i_pos = i_k_pos[0]
        k_pos = i_k_pos[1]
        for ii in range(0,len(i_pos)):
            clusterA[i_pos[ii],k_pos[ii]] = pp
            
        x_list.extend(i_pos)
        z_list.extend(k_pos)
        x_list.pop(0)
        z_list.pop(0)
    return clusterA

def avaiable_position(array, clusterA,diffzml, diffxml):
    xlist =[]
    for x in range(0, diffzml):
        for z in range(0, diffxml):
            val_new = position_check_A(x,z,clusterA)
            val_old = position_check_ini(x,z,array)
            if val_old == 1 and val_new == 0:
                xlist.append(x)
    if len(xlist) > 0:
        return 1
    else:
        return 0
    
def clustering(array,diffzml,diffxml):
    clusterA = np.zeros((diffzml,diffxml))
    for pp in range(100,500):
        w = avaiable_position(array, clusterA,diffzml, diffxml)
        if w == 1:
            cluster_1(array,diffzml,diffxml,clusterA, pp)
        else: 
            break
    return clusterA

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

def area(array,diffzml,diffxml, rr):
    count = 0
    for ii in range(0,diffzml):
        for kk in range(0,diffxml):
            if array[ii,kk] == rr:
                count+=1
    return count



def cluster_reduction(cluster_array,array, ll, diffzml, diffxml):
    clusterA = np.zeros((diffzml,diffxml))
    for ii in range(0,diffzml):
        for kk in range(0,diffxml):
            if cluster_array[ii,kk] == ll:
                clusterA[ii,kk] = array[ii,kk]
    return clusterA



def intial(image,x1,x2,z1,z2,UB,LB):
    transfer = image_position(image,x1,x2,z1,z2,UB,LB)
    diffzml = transfer[0]
    diffxml = transfer[1]
    cluster = clustering(transfer[2], diffzml,diffxml)
    d = int(np.max(cluster))
    pos = np.zeros((1,d))
    for rr in range(1,d):
        a = area(cluster,diffzml,diffxml, rr)
        pos[:,rr] = a
    ll = np.argmax(pos)
    LMLO = cluster_reduction(cluster,transfer[2], ll, diffzml, diffxml)

    return LMLO, diffzml, diffxml

image1 = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\dicom_convert\0145-1.png"
image2 = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\dicom_convert\0145-2.png"

LMLO = intial(image1,1818,1949,1436,1565,UB,LB)
CC = intial(image2,1372,1551,1492,1674,UB,LB)

def dis_check(array, diffzml, diffxml, input_size):
    New_image = np.zeros((input_size, input_size))
    discheck1 = np.zeros((1, diffxml))
    discheck2 = np.zeros((1, diffxml))
    for kk in range(0, diffzml):
        for ii in range(0,diffxml):
            if array[kk,ii] > 0:
                discheck1[:,ii] = ii
                break
            
    for kk in reversed(range(0, diffzml)):
        for ii in reversed(range(0,diffxml)):
            if array[kk,ii] > 0:
                discheck2[:,ii] = ii
                break
      
    minival = np.min(discheck1[np.nonzero(discheck1)])
    maxival = np.max(discheck2[np.nonzero(discheck2)])
    half = round(input_size/2)
    diff = maxival - minival
    middle = round(diff/2)
    image_centre = minival + middle
    ss = int(half - image_centre)
    print(ss)
    for kk in range(0, diffzml):
        for ii in range(0,diffxml):
            New_image[kk,ii+ss] = array[kk,ii]
    return New_image


im_LMLO = dis_check(LMLO[0], LMLO[1], LMLO[2], input_size)

im_CC = dis_check(CC[0], CC[1], CC[2], input_size)

plt.imshow(LMLO[0])

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




