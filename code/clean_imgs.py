# to make rotated images right as in annotation they were upright
import cv2
import glob
import os

img_root_path = '/home/sawyer/projects/trash_picker/data/images_png/'

img_name_list = glob.glob(os.path.join(img_root_path, '*.png'))
img_name_list.sort()

for img_name in img_name_list:
    img = cv2.imread(img_name)
    if img.shape[0] > img.shape[1]:
        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(img_name, img_rotate_90_clockwise)
        print(img_name)
