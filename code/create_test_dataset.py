import os
import glob

# input image path (absolute, relative both are fine)
img_path = '../tmp/imgs_gripper'

img_path = os.path.abspath(img_path)
img_list = glob.glob(os.path.join(img_path,'*.jpg'))
img_list.sort()

with open('/' + os.path.join(os.path.join(*img_path.split('/')[:-1]), 'test_gripper.txt'), 'w') as f:
    for i in img_list:
        f.write(i + '\n')