import cv2
import numpy as np
import os
import csv

path = os.chdir(r"C:\Users\cory1\Documents\ultrasound-dataset\DatasetA\original")
i = 1
for file in os.listdir(path):
    new_file_name = '{}.png'.format(i)
    print(file)
    os.rename(file, new_file_name)
    i = i + 1