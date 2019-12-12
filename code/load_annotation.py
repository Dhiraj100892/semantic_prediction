# to laod the annotation created using VGG via-2.0.8 saved in JSON file

import json
import os
import cv2
from IPython import embed
import numpy as np

img_root_path = '../data/img/top_camera'
out_path = '../data/label/top_camera'
annot_file = '../annot/top_camera_annot.json'

if not os.path.isdir(out_path):
    os.makedirs(out_path)

with open(annot_file) as f:
    data = json.load(f)

for count, key in enumerate(data['_via_img_metadata'].keys()):
    x = data['_via_img_metadata'][key]['regions'][0]['shape_attributes']['all_points_x']
    y = data['_via_img_metadata'][key]['regions'][0]['shape_attributes']['all_points_y']
    filename = data['_via_img_metadata'][key]['filename']
    org_img = cv2.imread(os.path.join(img_root_path, filename))
    annot_img = np.zeros((org_img.shape[0], org_img.shape[1]))
    cv2.fillPoly(annot_img, pts=[np.array(zip(x,y))], color=(255, 255, 255))
    cv2.imwrite(os.path.join(out_path, filename[:-3] + 'png'), annot_img)
    print(" img file = {} count = {}".format(filename, count+1))