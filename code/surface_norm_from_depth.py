#python surface_norm_from_depth.py
import numpy as np
from PIL import Image
import cv2
import os
import argparse


def get_normal_data(**args):
    """
    
    :param args: 
    :return: 
    """
    root = args['root_path']
    if not os.path.isdir(os.path.join(root, 'normal')):
        os.makedirs(os.path.join(root, 'normal'))
        
    # do it for both train & val
    txt_files = ['train_data','val_data']
    for txt_file in txt_files:
        lines = open(os.path.join(root, txt_file + '.txt'), 'r').readlines()
        
        for i in lines:
            word = i.split(' ')
        
            d_im = Image.open(word[1])
            d_im = np.array(d_im).astype(np.float32)
        
            kernel = np.ones((5,5),np.float32)/25
            d_im_blur = cv2.filter2D(d_im, -1, kernel)
        
            d_im[d_im == 0.0] = d_im_blur[d_im == 0.0]
        
            # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
            # to reduce noise
            zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
            zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
        
            # zy, zx = np.gradient(d_im)
            normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
            n = np.linalg.norm(normal, axis=2)
            normal[:, :, 0] /= n
            normal[:, :, 1] /= n
            normal[:, :, 2] /= n
        
            # offset and rescale values to be in 0-255
            normal += 1
            normal /= 2
            normal *= 255
            normal = normal.astype(np.uint8)
            word_split = word[1].split('/')
            out_file = '/'.join(j for j in word_split[:-2]) + '/normal/' + word_split[-1]
            cv2.imwrite(out_file, normal)
    
    # generate normal files as well
    for txt_file in txt_files:
        lines = open(os.path.join(root, txt_file + '.txt'), 'r').readlines()
        out_data_file = open(os.path.join(root, txt_file + '_normal.txt'), 'w') 
        for i in lines:
            word = i.split(' ')[0]
            word_split = word.split('/')
            normal_file_name = '/'.join(j for j in word_split[:-2]) + '/normal/' + word_split[-1]
            out_data_file.write(i[:-1] + ' ' + normal_file_name + '\n')
        
        out_data_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default='../data', help='root directory containing data')
    args = parser.parse_args()
    get_normal_data(**args.__dict__)

