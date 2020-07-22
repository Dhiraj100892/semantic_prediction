import os
import cv2
import numpy as np
from IPython import embed

file_path = 'stack_prediction_0_2.txt'
out_path = '/private/home/dhirajgandhi/project/suction_challenge/stacking_test/transition_pred_0_2'

if not os.path.isdir(out_path):
    os.makedirs(out_path)

with open(file_path, 'r') as f:
    lines = f.readlines()

for indx, line in enumerate(lines):
    word = line[:-1].split(' ')
    if int(word[1]) == 0:
        # cp images
        cmd = 'cp {} '.format(word[0]) + os.path.join(out_path, '{:06d}.jpg'.format(indx))
        os.system(cmd)
    else:
        img = cv2.imread(word[0])
        temp_img = 255*np.ones_like(img)
        img = cv2.addWeighted(img, 0.5, temp_img, 0.5, 0)
        cv2.imwrite(os.path.join(out_path, '{:06d}.jpg'.format(indx)), img)
    print(indx)
