# to laod the annotation created using VGG via-2.0.8 saved in JSON file

import json
import os
import cv2
from IPython import embed
import numpy as np
import copy

img_root_path = '../../../data/img/top_camera'
out_path = '../../../data/label/top_camera'
out_hand_path = '../../../data/label_hand/top_camera'
annot_file = '../annot/top_camera_annot.json'

if not os.path.isdir(out_path):
    os.makedirs(out_path)

if not os.path.isdir(out_hand_path):
    os.makedirs(out_hand_path)


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

    # take out the only gripper part
    img = cv2.imread(os.path.join(out_path, filename[:-3] + 'png'), cv2.IMREAD_UNCHANGED)

    # find pixels corresponding to rod
    Y, X = np.nonzero(img)

    # check the centroid first
    #cv2.circle(img, (int(X.mean()), int(Y.mean())), radius=50, color=(0), thickness=25)

    img2 = copy.deepcopy(img)
    img2[:3 * img.shape[0] / 4][:] = 0

    # fit the line
    Y2, X2 = np.nonzero(img2)
    #cv2.circle(img, (int(X2.mean()), int(Y2.mean())), radius=50, color=(0), thickness=25)

    z = np.polyfit(np.array([Y.mean(), Y2.mean()]), np.array([X.mean(), X2.mean()]), 1)

    # find the points on one side of line find the tip points
    p = np.poly1d(z)

    pos_p = None
    neg_p = None
    for x, y in zip(X, Y):
        if x - p(y) > 0:
            # img[y][x] = 0
            if pos_p is None or pos_p['y'] > y:
                pos_p = {'x': x, 'y': y}
        else:
            if neg_p is None or neg_p['y'] > y:
                neg_p = {'x': x, 'y': y}

    #cv2.circle(img, (pos_p['x'], pos_p['y']), radius=50, color=(0), thickness=25)
    #cv2.circle(img, (neg_p['x'], neg_p['y']), radius=50, color=(0), thickness=25)

    img[min(pos_p['y'], neg_p['y']) + 425:][:] = 0
    #cv2.imshow('test', cv2.resize(img, (img.shape[1] / 8, img.shape[0] / 8)))
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_hand_path, filename[:-3] + 'png'), img)
