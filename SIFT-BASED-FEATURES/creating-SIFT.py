import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

def sift_features(image)
    Beginning_image = cv2.imread(image,0) #read image "0" for gray and "1" for colour

    def Image_check_x(image):
        p = image.shape[0]
        if ((p+3)/4) != int((p+3)/4):
            if ((p+2)/4) != int((p+2)/4):
                if ((p+1)/4) != int((p+1)/4):
                    return p
                else:
                    return p+1
            else:
                return p+2
        else:
            return p+3



    def Image_check_y(image):
        p = image.shape[1]
        if ((p+3)/4) != int((p+3)/4):
            if ((p+2)/4) != int((p+2)/4):
                if ((p+1)/4) != int((p+1)/4):
                    return p
                else:
                    return p+1
            else:
                return p+2
        else:
            return p+3
    
    
    q = int(Image_check_x(Beginning_image))
    u = int(Image_check_y(Beginning_image))
    beginning_image = cv2.resize(Beginning_image,(u,q))
    s = 3
    k = 2 ** (1.0 / s) #kernals
    sig = 1.6
    kvector = np.array([sig, k*sig, sig*(k**2), sig*(k**3), sig*(k**4), sig*(k**5),sig*(k**6), sig*(k**7), sig*(k**8)])
    c = -1
    threshold = 1

    def gauss_blur(sigma):

        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        size = 2*np.ceil(3*sigma)+1
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    #pyrlvl[0] = cv2.filter2D(Beginning_image,c, gauss_blur(s,k))
    #print (gauss_blur(8,5))
    #plt.imshow(filter_image_Gblur3, cmap ='gray')
    #cv2.imwrite('/home/cot12/Documents/jup-pads/sky-2.jpg', filter_image_Gblur1)
    # detA = (dxx*((dyy * dzz)-(dyz**2))) + (dxy*((dxy*dzz)-(dxz*dyz))) + (dxz*((dxy*dxz)-((dyy*dxz))))

    #create image size variant
    doubled = cv2.resize((beginning_image), ((2*u),(2*q)))
    normal = cv2.resize((doubled),(u,q))
    half = cv2.resize((normal), (int(u/2),int(q/2)))
    quarter = cv2.resize((half),(int(u/4),int(q/4)))

    #create zero layer for pyramid
    pyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1],6))
    pyrlvl2 = np.zeros((normal.shape[0], normal.shape[1],6))
    pyrlvl3 = np.zeros((half.shape[0], half.shape[1],6))
    pyrlvl4 = np.zeros((quarter.shape[0], quarter.shape[1],6))

    # create our Gaussian pyramid
    for i in range(0,6):
        pyrlvl1[:,:,i] = ndimage.filters.gaussian_filter(doubled, kvector[i]) 
        pyrlvl2[:,:,i] = ndimage.filters.gaussian_filter(normal, kvector[i+1])
        pyrlvl3[:,:,i] = ndimage.filters.gaussian_filter(half, kvector[i+2])
        pyrlvl4[:,:,i] = ndimage.filters.gaussian_filter(quarter,kvector[i+3])

    #create zero layer for DoG images 
    DoGlvl1 = np.zeros((doubled.shape[0], doubled.shape[1],5))
    DoGlvl2 = np.zeros((normal.shape[0], normal.shape[1],5))
    DoGlvl3 = np.zeros((half.shape[0], half.shape[1],5))
    DoGlvl4 = np.zeros((quarter.shape[0], quarter.shape[1],5))

    # create DoG layer
    for i in range (0,5):
        DoGlvl1[:,:,i] = pyrlvl1[:,:,i+1] - pyrlvl1[:,:,i]
        DoGlvl2[:,:,i] = pyrlvl2[:,:,i+1] - pyrlvl2[:,:,i]
        DoGlvl3[:,:,i] = pyrlvl3[:,:,i+1] - pyrlvl3[:,:,i]
        DoGlvl4[:,:,i] = pyrlvl4[:,:,i+1] - pyrlvl4[:,:,i]
    
    # create zero layer to store extrema location
    Exlvl1 = np.zeros((doubled.shape[0], doubled.shape[1],3))
    Exlvl2 = np.zeros((normal.shape[0], normal.shape[1],3))
    Exlvl3 = np.zeros((half.shape[0], half.shape[1],3))
    Exlvl4 = np.zeros((quarter.shape[0], quarter.shape[1],3))

    for i in range(1,4):
        for j in range(16, doubled.shape[0] -16):
            for k in range(16, doubled.shape[1]-16):
                if np.absolute(DoGlvl1[j, k, i]) < threshold:
                    continue
                maxima = DoGlvl1[j,k,i] > 0
                minima = DoGlvl1[j,k,i] < 0
                for di in range(-1,2):
                    for dj in range(-1,2):
                        for dk in range(-1,2):
                            if di == 0 and dj ==0 and dk == 0:
                                continue
                            maxima = maxima and (DoGlvl1[j, k, i] > DoGlvl1[j + dj, k + dk, i + di])
                            minima = minima and (DoGlvl1[j, k, i] < DoGlvl1[j + dj, k + dk, i + di])
                        
                            if not maxima and not minima:
                                break

                        if not maxima and not minima:
                            break

                    if not maxima and not minima:
                        break
                if maxima or minima:
                    dx = (DoGlvl1[j, k+1, i] - DoGlvl1[j, k-1, i]) * 0.5 / 255
                    dy = (DoGlvl1[j+1, k, i] - DoGlvl1[j-1, k, i]) * 0.5 / 255
                    dz = (DoGlvl1[j, k, i+1] - DoGlvl1[j, k, i-1]) * 0.5 / 255
                    dxx = (DoGlvl1[j, k+1, i] + DoGlvl1[j, k-1, i] - 2 * DoGlvl1[j, k, i]) * 1.0 / 255        
                    dyy = (DoGlvl1[j+1, k, i] + DoGlvl1[j-1, k, i] - 2 * DoGlvl1[j, k, i]) * 1.0 / 255          
                    dzz = (DoGlvl1[j, k, i+1] + DoGlvl1[j, k, i-1] - 2 * DoGlvl1[j, k, i]) * 1.0 / 255
                    dxy = (DoGlvl1[j+1, k+1, i] - DoGlvl1[j+1, k-1, i] - DoGlvl1[j-1, k+1, i] + DoGlvl1[j-1, k-1, i]) * 0.25 / 255 
                    dxz = (DoGlvl1[j, k+1, i+1] - DoGlvl1[j, k-1, i+1] - DoGlvl1[j, k+1, i-1] + DoGlvl1[j, k-1, i-1]) * 0.25 / 255 
                    dyz = (DoGlvl1[j+1, k, i+1] - DoGlvl1[j-1, k, i+1] - DoGlvl1[j+1, k, i-1] + DoGlvl1[j-1, k, i-1]) * 0.25 / 255  
                
                
                    J = np.matrix([[dx], [dy], [dz]])
                    DH = np.matrix([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
                    detA = (dxx*((dyy * dzz)-(dyz**2))) + (dxy*((dxy*dzz)-(dxz*dyz))) + (dxz*((dxy*dxz)-((dyy*dxz))))
                    invDH = np.linalg.pinv(DH)
                
                    X_hat = np.dot(invDH,J)
                    D_X_hat = DoGlvl1[j,k,i] + (0.5 * np.dot(J,np.transpose(X_hat))) #contast variable
                    r = 10.0
                    if [(((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(D_X_hat) < 0.03]:
                        Exlvl1[j, k, i - 1] = 1

    for i in range(1,4):
        for j in range(8, normal.shape[0] -8):
            for k in range(8, normal.shape[1]-8):
                if np.absolute(DoGlvl2[j, k, i]) < threshold:
                    continue
                maxima = DoGlvl2[j,k,i] > 0
                minima = DoGlvl2[j,k,i] < 0
                for di in range(-1,2):
                    for dj in range(-1,2):
                        for dk in range(-1,2):
                            if di == 0 and dj ==0 and dk == 0:
                                continue
                            maxima = maxima and (DoGlvl2[j, k, i] > DoGlvl2[j + dj, k + dk, i + di])
                            minima = minima and (DoGlvl2[j, k, i] < DoGlvl2[j + dj, k + dk, i + di])
                        
                            if not maxima and not minima:
                                break

                        if not maxima and not minima:
                            break

                    if not maxima and not minima:
                        break
                if maxima or minima:
                    dx = (DoGlvl2[j, k+1, i] - DoGlvl2[j, k-1, i]) * 0.5 / 255
                    dy = (DoGlvl2[j+1, k, i] - DoGlvl2[j-1, k, i]) * 0.5 / 255
                    dz = (DoGlvl2[j, k, i+1] - DoGlvl2[j, k, i-1]) * 0.5 / 255
                    dxx = (DoGlvl2[j, k+1, i] + DoGlvl2[j, k-1, i] - 2 * DoGlvl2[j, k, i]) * 1.0 / 255        
                    dyy = (DoGlvl2[j+1, k, i] + DoGlvl2[j-1, k, i] - 2 * DoGlvl2[j, k, i]) * 1.0 / 255          
                    dzz = (DoGlvl2[j, k, i+1] + DoGlvl2[j, k, i-1] - 2 * DoGlvl2[j, k, i]) * 1.0 / 255
                    dxy = (DoGlvl2[j+1, k+1, i] - DoGlvl2[j+1, k-1, i] - DoGlvl2[j-1, k+1, i] + DoGlvl2[j-1, k-1, i]) * 0.25 / 255 
                    dxz = (DoGlvl2[j, k+1, i+1] - DoGlvl2[j, k-1, i+1] - DoGlvl2[j, k+1, i-1] + DoGlvl2[j, k-1, i-1]) * 0.25 / 255 
                    dyz = (DoGlvl2[j+1, k, i+1] - DoGlvl2[j-1, k, i+1] - DoGlvl2[j+1, k, i-1] + DoGlvl2[j-1, k, i-1]) * 0.25 / 255  
                
                
                    J = np.matrix([[dx], [dy], [dz]])
                    DH = np.matrix([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
                    detA = (dxx*((dyy * dzz)-(dyz**2))) + (dxy*((dxy*dzz)-(dxz*dyz))) + (dxz*((dxy*dxz)-((dyy*dxz))))
                    invDH = np.linalg.pinv(DH)
                    X_hat = np.dot(invDH,J)
                    D_X_hat = DoGlvl2[j,k,i] + (0.5 * np.dot(J,np.transpose(X_hat))) #contast variable
                    r = 10.0
                    if [(((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(D_X_hat) < 0.03]:
                        Exlvl2[j, k, i - 1] = 1
                    
    for i in range(1,4):
        for j in range(4, half.shape[0] -4):
            for k in range(4, half.shape[1]-4):
                if np.absolute(DoGlvl3[j, k, i]) < threshold:
                    continue
                maxima = DoGlvl3[j,k,i] > 0
                minima = DoGlvl3[j,k,i] < 0
                for di in range(-1,2):
                    for dj in range(-1,2):
                        for dk in range(-1,2):
                            if di == 0 and dj ==0 and dk == 0:
                                continue
                            maxima = maxima and (DoGlvl3[j, k, i] > DoGlvl3[j + dj, k + dk, i + di])
                            minima = minima and (DoGlvl3[j, k, i] < DoGlvl3[j + dj, k + dk, i + di])
                        
                            if not maxima and not minima:
                                break

                        if not maxima and not minima:
                            break

                    if not maxima and not minima:
                        break
                if maxima or minima:
                    dx = (DoGlvl3[j, k+1, i] - DoGlvl3[j, k-1, i]) * 0.5 / 255
                    dy = (DoGlvl3[j+1, k, i] - DoGlvl3[j-1, k, i]) * 0.5 / 255
                    dz = (DoGlvl3[j, k, i+1] - DoGlvl3[j, k, i-1]) * 0.5 / 255
                    dxx = (DoGlvl3[j, k+1, i] + DoGlvl3[j, k-1, i] - 2 * DoGlvl3[j, k, i]) * 1.0 / 255        
                    dyy = (DoGlvl3[j+1, k, i] + DoGlvl3[j-1, k, i] - 2 * DoGlvl3[j, k, i]) * 1.0 / 255          
                    dzz = (DoGlvl3[j, k, i+1] + DoGlvl3[j, k, i-1] - 2 * DoGlvl3[j, k, i]) * 1.0 / 255
                    dxy = (DoGlvl3[j+1, k+1, i] - DoGlvl3[j+1, k-1, i] - DoGlvl3[j-1, k+1, i] + DoGlvl3[j-1, k-1, i]) * 0.25 / 255 
                    dxz = (DoGlvl3[j, k+1, i+1] - DoGlvl3[j, k-1, i+1] - DoGlvl3[j, k+1, i-1] + DoGlvl3[j, k-1, i-1]) * 0.25 / 255 
                    dyz = (DoGlvl3[j+1, k, i+1] - DoGlvl3[j-1, k, i+1] - DoGlvl3[j+1, k, i-1] + DoGlvl3[j-1, k, i-1]) * 0.25 / 255  
                
                
                    J = np.matrix([[dx], [dy], [dz]])
                    DH = np.matrix([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
                    detA = (dxx*((dyy * dzz)-(dyz**2))) + (dxy*((dxy*dzz)-(dxz*dyz))) + (dxz*((dxy*dxz)-((dyy*dxz))))
                    invDH = np.linalg.pinv(DH)
                
                    X_hat = np.dot(invDH,J)
                    D_X_hat = DoGlvl3[j,k,i] + (0.5 * np.dot(J,np.transpose(X_hat))) #contast variable
                    r = 10.0
                    if [(((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(D_X_hat) < 0.03]:
                        Exlvl3[j, k, i - 1] = 1
        
    for i in range(1,4):
        for j in range(2, quarter.shape[0] -2):
            for k in range(2, quarter.shape[1]-2):
                if np.absolute(DoGlvl4[j, k, i]) < threshold:
                    continue
                maxima = DoGlvl4[j,k,i] > 0
                minima = DoGlvl4[j,k,i] < 0
                for di in range(-1,2):
                    for dj in range(-1,2):
                        for dk in range(-1,2):
                            if di == 0 and dj ==0 and dk == 0:
                                continue
                            maxima = maxima and (DoGlvl4[j, k, i] > DoGlvl4[j + dj, k + dk, i + di])
                            minima = minima and (DoGlvl4[j, k, i] < DoGlvl4[j + dj, k + dk, i + di])
                        
                            if not maxima and not minima:
                                break

                        if not maxima and not minima:
                            break

                    if not maxima and not minima:
                        break
                if maxima or minima:
                    dx = (DoGlvl4[j, k+1, i] - DoGlvl4[j, k-1, i]) * 0.5 / 255
                    dy = (DoGlvl4[j+1, k, i] - DoGlvl4[j-1, k, i]) * 0.5 / 255
                    dz = (DoGlvl4[j, k, i+1] - DoGlvl4[j, k, i-1]) * 0.5 / 255
                    dxx = (DoGlvl4[j, k+1, i] + DoGlvl4[j, k-1, i] - 2 * DoGlvl4[j, k, i]) * 1.0 / 255        
                    dyy = (DoGlvl4[j+1, k, i] + DoGlvl4[j-1, k, i] - 2 * DoGlvl4[j, k, i]) * 1.0 / 255          
                    dzz = (DoGlvl4[j, k, i+1] + DoGlvl4[j, k, i-1] - 2 * DoGlvl4[j, k, i]) * 1.0 / 255
                    dxy = (DoGlvl4[j+1, k+1, i] - DoGlvl4[j+1, k-1, i] - DoGlvl4[j-1, k+1, i] + DoGlvl4[j-1, k-1, i]) * 0.25 / 255 
                    dxz = (DoGlvl4[j, k+1, i+1] - DoGlvl4[j, k-1, i+1] - DoGlvl4[j, k+1, i-1] + DoGlvl4[j, k-1, i-1]) * 0.25 / 255 
                    dyz = (DoGlvl4[j+1, k, i+1] - DoGlvl4[j-1, k, i+1] - DoGlvl4[j+1, k, i-1] + DoGlvl4[j-1, k, i-1]) * 0.25 / 255  
                
                
                    J = np.matrix([[dx], [dy], [dz]])
                    DH = np.matrix([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
                    detA = (dxx*((dyy * dzz)-(dyz**2))) + (dxy*((dxy*dzz)-(dxz*dyz))) + (dxz*((dxy*dxz)-((dyy*dxz))))
                    invDH = np.linalg.pinv(DH)
                
                    X_hat = np.dot(invDH,J)
                    D_X_hat = DoGlvl4[j,k,i] + (0.5 * np.dot(J,np.transpose(X_hat))) #contast variable
                    r = 10.0
                    if [(((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(D_X_hat) < 0.03]:
                        Exlvl4[j, k, i - 1] = 1
                    
    print("Number of extrema in first octave: %d" % np.sum(Exlvl1))
    print("Number of extrema in second octave: %d" % np.sum(Exlvl2))
    print("Number of extrema in third octave: %d" % np.sum(Exlvl3))
    print("Number of extrema in fourth octave: %d" % np.sum(Exlvl4))

    extr_sum = int(np.sum(Exlvl1)) + int(np.sum(Exlvl2)) + int(np.sum(Exlvl3)) + int(np.sum(Exlvl4))

    keypoints = np.zeros((extr_sum, 4)) 

    # Gradient magnitude and orientation for each image sample point at each scale
    Grmaglvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    Grmaglvl2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    Grmaglvl3 = np.zeros((half.shape[0], half.shape[1], 3))
    Grmaglvl4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    Orientlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    Orientlvl2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    Orientlvl3 = np.zeros((half.shape[0], half.shape[1], 3))
    Orientlvl4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    for i in range(0,3):
        for j in range(1, doubled.shape[0]-1):
            for k in range(1, doubled.shape[1]-1):
                Grmaglvl1[j,k,i] = (((doubled[j+1,k] - doubled[j-1,k])**2) + ((doubled[j,k+1] - doubled[j,k-1])**2))**0.5
                Orientlvl1[j,k,i] =(((180/np.pi)*(np.pi + np.arctan2((doubled[j,k+1] - doubled[j,k-1]), (doubled[j+1,k] - doubled[j-1,k])))))

    for i in range(0,3):
        for j in range(1, normal.shape[0]-1):
            for k in range(1, normal.shape[1]-1):
                Grmaglvl2[j,k,i] = (((normal[j+1,k] - normal[j-1,k])**2) + ((normal[j,k+1] - normal[j,k-1])**2))**0.5
                Orientlvl2[j,k,i] =((180/np.pi)*(np.pi + np.arctan2((normal[j,k+1] - normal[j,k-1]), (normal[j+1,k] - normal[j-1,k]))))
            
    for i in range(0,3):
        for j in range(1, half.shape[0]-1):
            for k in range(1, half.shape[1]-1):
                Grmaglvl3[j,k,i] = (((half[j+1,k] - half[j-1,k])**2) + ((half[j,k+1] - half[j,k-1])**2))**0.5
                Orientlvl3[j,k,i] =((180/np.pi)*(np.pi + np.arctan2((half[j,k+1] - half[j,k-1]), (half[j+1,k] - half[j-1,k]))))
            
    for i in range(0,3):
        for j in range(1, quarter.shape[0]-1):
            for k in range(1, quarter.shape[1]-1):
                Grmaglvl4[j,k, i] = (((quarter[j+1,k] - quarter[j-1,k])**2) + ((quarter[j,k+1] - quarter[j,k-1])**2))**0.5
                Orientlvl4[j,k,i] =((180/np.pi)*(np.pi + np.arctan2((quarter[j,k+1] - quarter[j,k-1]), (quarter[j+1,k]- quarter[j-1,k]))))
    
    print("Calculating keypoint orientations...")

    def quantize_orientation(theta, num_bins): 
      bin_width = 360//num_bins 
      return int(np.floor(theta)//bin_width)

    count = 0
    num_bins = 36

    for i in range(0, 3):
            for j in range(0, doubled.shape[0] - 1):
                for k in range(0, doubled.shape[1] - 1):
                    if Exlvl1[j, k, i] == 1:
                        new_sig = 1.5*kvector[i]
                        w = int(2*np.ceil(new_sig)+1)
                        gaussian_window = gauss_blur(new_sig)
                        orient_histogram = np.zeros(num_bins,dtype=np.float32)
                        for x in range (-w, w+1):
                            for y in range (-w, w+1):
                                if x+j<0 or x+j> doubled.shape[0]-1:
                                    continue
                                if y+k<0 or y+k> doubled.shape[1]-1:
                                    continue
                                weight = Grmaglvl1[j+x,k+y, i]* gaussian_window[x,y]
                                theta = Orientlvl1[j,k,i]
                                bin = quantize_orientation(theta, num_bins)
                                orient_histogram[bin] += weight
                        maxvalue = np.amax(orient_histogram)
                        maxindex = np.argmax(orient_histogram)
                        keypoints[count, :] = np.array([[int(k/2) , int(j/2) , kvector[i], maxindex]])
                        count+= 1
                    
    for i in range(0, 3):
            for j in range(0, normal.shape[0] - 1):
                for k in range(0, normal.shape[1] - 1):
                    if Exlvl2[j, k, i] == 1:
                        new_sig = 1.5*kvector[i+1]
                        w = int(2*np.ceil(new_sig)+1)
                        gaussian_window = gauss_blur(new_sig)
                        orient_histogram = np.zeros(num_bins,dtype=np.float32)
                        for x in range (-w, w+1):
                            for y in range (-w, w+1):
                                if x+j<0 or x+j> normal.shape[0]-1:
                                    continue
                                if y+k<0 or y+k> normal.shape[1]-1:
                                    continue
                                weight = Grmaglvl2[j+x,k+y, i]* gaussian_window[x,y]
                                theta = Orientlvl2[j,k,i]
                                bin = quantize_orientation(theta, num_bins)
                                orient_histogram[bin] += weight
                        maxvalue = np.amax(orient_histogram)
                        maxindex = np.argmax(orient_histogram)
                        keypoints[count, :] = np.array([[int(k) , int(j) , kvector[i+1], maxindex]])
                        count+= 1
                    
    for i in range(0, 3):
            for j in range(0, half.shape[0] - 1):
                for k in range(0, half.shape[1] - 1):
                    if Exlvl3[j, k, i] == 1:
                        new_sig = 1.5*kvector[i+2]
                        w = int(2*np.ceil(new_sig)+1)
                        gaussian_window = gauss_blur(new_sig)
                        orient_histogram = np.zeros(num_bins,dtype=np.float32)
                        for x in range (-w, w+1):
                            for y in range (-w, w+1):
                                if x+j<0 or x+j> half.shape[0]-1:
                                    continue
                                if y+k<0 or y+k> half.shape[1]-1:
                                    continue
                                weight = Grmaglvl3[j+x,k+y, i]* gaussian_window[x,y]
                                theta = Orientlvl3[j,k,i]
                                bin = quantize_orientation(theta, num_bins)
                                orient_histogram[bin] += weight
                        maxvalue = np.amax(orient_histogram)
                        maxindex = np.argmax(orient_histogram)
                        keypoints[count, :] = np.array([[int(2*k) , int(2*j) , kvector[i+2], maxindex]])
                        count+= 1
                    
    for i in range(0, 3):
            for j in range(0, quarter.shape[0] - 1):
                for k in range(0, quarter.shape[1] - 1):
                    if Exlvl4[j, k, i] == 1:
                        new_sig = 1.5*kvector[i+3]
                        w = int(2*np.ceil(new_sig)+1)
                        gaussian_window = gauss_blur(new_sig)
                        orient_histogram = np.zeros(num_bins,dtype=np.float32)
                        for x in range (-w, w+1):
                            for y in range (-w, w+1):
                                if x+j<0 or x+j> quarter.shape[0]-1:
                                    continue
                                if y+k<0 or y+k> quarter.shape[1]-1:
                                    continue
                                weight = Grmaglvl4[j+x,k+y, i]* gaussian_window[x,y]
                                theta = Orientlvl4[j,k,i]
                                bin = quantize_orientation(theta, num_bins)
                                orient_histogram[bin] += weight
                        maxvalue = np.amax(orient_histogram)
                        maxindex = np.argmax(orient_histogram)
                        keypoints[count, :] = np.array([[int(4*k) , int(4*j) , kvector[i+3], maxindex]])
                        count+= 1
                    
                            
    print("Calculating descriptor...")
    magnitude = np.zeros((normal.shape[0], normal.shape[1], 12))
    orientation = np.zeros((normal.shape[0], normal.shape[1], 12))
    for i in range (0,3):
        magmax = np.amax(Grmaglvl1[:, :, i])
        magnitude[:, :, i] = cv2.resize(Grmaglvl1[:, :, i],None, fx = 0.5 , fy = 0.5).astype(float)
        magnitude[:, :, i] = (magmax/np.amax(magnitude[:, :, i]))* magnitude[:, :, i]
        orientation[:, :, i] = cv2.resize(Orientlvl1[:, :, i],None, fx = 0.5 , fy = 0.5).astype(int)
        orientation[:, :, i] = ((36//np.amax(orientation[:, :, i]))*orientation[:, :, i]).astype(int)
    for i in range(0,3):
        magnitude[:, :, i+3] = Grmaglvl2[:, :, i].astype(float)
        orientation[:, :, i+3] = Orientlvl2[:, :, i].astype(int)
    for i in range(0,3):
        magnitude[:, :, i+6] = cv2.resize(Grmaglvl3[:, :, i],None, fx = 2 , fy = 2).astype(int)
        orientation[:, :, i+6] = cv2.resize(Orientlvl3[:, :, i],None, fx = 2 , fy = 2).astype(int)
    for i in range(0,3):
        magnitude[:, :, i+9] = cv2.resize(Grmaglvl4[:, :, i],None, fx = 4 , fy = 4).astype(float)
        orientation[:, :, i+9] = cv2.resize(Orientlvl4[:, :, i],None, fx = 4 , fy = 4).astype(int)
    
    np.savetxt('/impacs/cot12/bin/descriptors1', descriptors, delimiter=',', fmt='%d')
    
    def image-print(image2)
            radius = 3
            thickness = 1
            color = (255, 255, 255)
            for i in range(0, int(keypoints.shape[0])):
                center_coordinates = (int(keypoints[i,0]),int(keypoints[i,1]))
                cv2.circle(image2, center_coordinates, radius, color, thickness)
            cv2.imwrite('/impacs/cot12/bin/features-boob.jpg',image2)
    
    image-print(image)
image = '/impacs/cot12/bin/boob-2.jpg'     
sift_features(image)
