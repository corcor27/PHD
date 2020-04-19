import numpy as np
import cv2
import pydicom as dicom
from skimage import exposure
import matplotlib.pyplot as plt
import os
import csv
from scipy import ndimage

def random_field(array):
    


N = 800
ran_array = np.round(np.random.random((N,N,N))*100)

