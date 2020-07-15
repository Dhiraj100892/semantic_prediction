# sample 100 images from dataset
import random
import os

num_img = 100
txt_file = '/private/home/dhirajgandhi/project/suction_challenge/test_data/imgs/test_gripper.txt'
out_dir = '../sampled_data'

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

with open(txt_file, 'r') as f:
    lines = random.sample(f.readlines(), num_img)

for count, line in enumerate(lines):
    cmd = 'cp {} {}/{:04d}.jpg'.format(line[:-1], out_dir, count)
    print(cmd)
    os.system(cmd)


