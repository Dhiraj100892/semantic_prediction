# take out the bounding box around the gripper based on the hand coded pixel location
import cv2
import glob
from IPython import embed
import os

img_name_list = glob.glob('../../../data/img/top_camera/*.JPG')
img_name_list.sort()
label_name_list = glob.glob('../../../data/label/top_camera/*.png')
label_name_list.sort()

out_img_path = '../../../data/img_crop_hand/top_camera'
out_label_path = '../../../data/label_crop_hand/top_camera'

if not os.path.isdir(out_img_path):
    os.makedirs(out_img_path)

if not os.path.isdir(out_label_path):
    os.makedirs(out_label_path)

for img_name, label_name in zip(img_name_list, label_name_list):
    img = cv2.imread(img_name)
    label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
    mean_pt = [img.shape[0]/2 + img.shape[0]/20, img.shape[1]/2 + img.shape[0]/25]
    box_size = img.shape[0]/3
    img = img[mean_pt[0]-box_size/2:mean_pt[0]+box_size/2, mean_pt[1]-box_size/2:mean_pt[1]+box_size/2]
    label = label[mean_pt[0]-box_size/2:mean_pt[0]+box_size/2, mean_pt[1]-box_size/2:mean_pt[1]+box_size/2]
    '''
    cv2.imshow('test', cv2.resize(img, (img.shape[1]/8, img.shape[0]/8)))
    cv2.waitKey(0)
    '''
    print(os.path.join(out_img_path, img_name.split('/')[-1]))
    cv2.imwrite(os.path.join(out_img_path, img_name.split('/')[-1]), img)
    cv2.imwrite(os.path.join(out_label_path, label_name.split('/')[-1]), label)
