# take out the bounding box around the gripper based on the hand coded pixel location
import cv2
import glob
from IPython import embed
import os

img_name_list = glob.glob('../tmp/imgs/*.jpg')
img_name_list.sort()

out_img_path = '../tmp/imgs_gripper'

if not os.path.isdir(out_img_path):
    os.makedirs(out_img_path)

for img_name in img_name_list:
    img = cv2.imread(img_name)

    mean_pt = [int(img.shape[0]/2 + img.shape[0]/20), int(img.shape[1]/2 + img.shape[0]/25)]
    box_size = int(img.shape[0]/2)
    img = img[int(mean_pt[0]-box_size/2):int(mean_pt[0]+box_size/2), int(mean_pt[1]-box_size/2):int(mean_pt[1]+box_size/2)]
    '''
    cv2.imshow('test', cv2.resize(img, (img.shape[1]/8, img.shape[0]/8)))
    cv2.waitKey(0)
    '''
    print(os.path.join(out_img_path, img_name.split('/')[-1]))
    cv2.imwrite(os.path.join(out_img_path, img_name.split('/')[-1]), img)