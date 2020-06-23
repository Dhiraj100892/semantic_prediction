# code to figure out the distance between the finger tip points given the mask of trash picker
import numpy as np
import cv2
from IPython import embed
import copy
import glob

img_name_list = glob.glob('../data/label/top_camera/*.png')

for img_name in img_name_list:

    img = cv2.imread(img_name ,cv2.IMREAD_UNCHANGED)

    # find pixels corresponding to rod
    Y, X = np.nonzero(img)

    # check the centroid first
    cv2.circle(img, (int(X.mean()), int(Y.mean())), radius=50, color=(0), thickness=25)

    img2 = copy.deepcopy(img)
    img2[:3*img.shape[0]/4][:] = 0

    # fit the line
    Y2, X2 = np.nonzero(img2)
    cv2.circle(img, (int(X2.mean()), int(Y2.mean())), radius=50, color=(0), thickness=25)

    z = np.polyfit(np.array([Y.mean(),Y2.mean()]), np.array([X.mean(),X2.mean()]),1)

    # find the points on one side of line find the tip points
    p = np.poly1d(z)

    pos_p = None
    neg_p = None
    for x, y in zip(X, Y):
        if x - p(y) > 0:
            #img[y][x] = 0
            if pos_p is None or pos_p['y'] > y:
                pos_p = {'x':x, 'y':y}
        else:
            if neg_p is None or neg_p['y'] > y:
                neg_p = {'x':x, 'y':y}
    
    cv2.circle(img, (pos_p['x'], pos_p['y']), radius=50, color=(0), thickness=25)
    cv2.circle(img, (neg_p['x'], neg_p['y']), radius=50, color=(0), thickness=25)

    img[min(pos_p['y'], neg_p['y'])+425:][:] = 0
    cv2.imshow('test',cv2.resize(img, (img.shape[1]/8, img.shape[0]/8)))
    cv2.waitKey(0)


# calculate the distance