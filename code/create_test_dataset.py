import os
import glob

# input image path (absolute, relative both are fine)
img_path = '/private/home/dhirajgandhi/project/suction_challenge/test_data/imgs'

img_path = os.path.abspath(img_path)

with open(os.path.join(img_path, 'test_gripper.txt'), 'w') as f:
    for d in os.listdir(img_path):
        img_list = glob.glob(os.path.join(img_path, d, '*.jpg'))
        img_list.sort()
        for i in img_list:
            f.write(i + '\n')