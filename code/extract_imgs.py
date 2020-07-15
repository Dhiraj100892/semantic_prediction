import os
import glob

root_dir = '/private/home/dhirajgandhi/project/suction_challenge/test_data'
video_dir = ['videos']

out_dir = 'imgs'

for v in video_dir:
    videos = glob.glob(os.path.join(root_dir, v, '*.MP4'))

    for vi in videos:

        # mkdir for imgs
        img_dir = os.path.join(root_dir, out_dir, vi.split('/')[-1].split('.')[0])
        print(img_dir)
        os.makedirs(img_dir)

        # extract images
        cmd =  'ffmpeg -i {}  {}/%04d.jpg'.format(vi, img_dir)
        print(cmd)
        os.system(cmd)