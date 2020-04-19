import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt1
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.mplot3d import Axes3D
file = r"C:\Users\cory1\Documents\test-folder\data_set\Malignant\output\1467-2.dcm.npy"
dimesional_array = np.load(file)
diffzml = dimesional_array.shape[0]
diffxml = dimesional_array.shape[1]

plt.imshow(dimesional_array)