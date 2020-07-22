import cv2
import numpy as np
from IPython import embed

file_name = '../data_v_2.txt'

with open(file_name, 'r') as f:
    lines = f.readlines()

mean_img = None
for line in lines:
    annot = cv2.imread(line.split(' ')[-1][:-1], cv2.IMREAD_UNCHANGED)
    annot = cv2.resize(annot, (480,480)) 
    if mean_img is None:
        mean_img = np.zeros_like(annot).astype(np.float32)
    mean_img += (annot / 255)

mean_img[mean_img != 0] = 255 
mean_img = mean_img.astype(np.uint8)
cv2.imwrite('mean_img.png',mean_img)