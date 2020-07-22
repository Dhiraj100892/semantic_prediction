import glob
import os

path = '/private/home/dhirajgandhi/project/suction_challenge/stacking_test/test_stacking_policy'
out_file = open('stacking_test.txt', 'w')
img_list = []

for d in os.listdir(path):
    img_list += glob.glob(os.path.join(path, d, 'images', '*.png'))

img_list.sort()
for l in img_list:
    out_file.write(l + '\n')

out_file.close()

