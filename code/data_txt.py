# python data_txt.py
import glob
import random
import argparse
from IPython import embed


def enumerate_data(**args):
    """
    Enumerate the data available and generates the train and val files out of it
    :param args:
    :return:
    """
    root = args['root_path']
    color_data = glob.glob('/private/home/dhirajgandhi/project/suction_challenge/data/img_crop_hand/top_camera/*.JPG') + glob.glob('/private/home/dhirajgandhi/project/suction_challenge/sampled_data_cropped/*.jpg')
    label_data = glob.glob('/private/home/dhirajgandhi/project/suction_challenge/data/label_crop_hand/top_camera/*.png') + glob.glob('/private/home/dhirajgandhi/project/suction_challenge/sampled_data_cropped_annotated/*.png')

    color_data.sort() 
    label_data.sort()

    out_file = open('/private/home/dhirajgandhi/project/suction_challenge/data_v_2.txt', 'w')
    train_out_file = open('/private/home/dhirajgandhi/project/suction_challenge/train_data_v_2.txt', 'w')
    val_out_file = open( '/private/home/dhirajgandhi/project/suction_challenge/val_data_v_2.txt', 'w')
    train_ratio = 0.9

    for i in range(len(color_data)):
        out_file.write(color_data[i] + ' ' + label_data[i] + '\n')
        if random.random() < train_ratio:
            train_out_file.write(color_data[i] + ' ' + label_data[i] + '\n')
        else:
            val_out_file.write(color_data[i] + ' ' + label_data[i] + '\n')

    out_file.close()
    train_out_file.close()
    val_out_file.close()
    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', default='../data', help='path to the folder containing data')
    args = parser.parse_args()
    enumerate_data(**args.__dict__)
