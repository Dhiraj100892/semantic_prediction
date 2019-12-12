# python data_txt.py
import glob
import random
import argparse


def enumerate_data(**args):
    """
    Enumerate the data available and generates the train and val files out of it
    :param args:
    :return:
    """
    root = args['root_path']
    color_data = glob.glob(root + '/img/top_camera/*.JPG')
    label_data = glob.glob(root + '/label/top_camera/*.png')

    out_file = open(root + '/data.txt', 'w')
    train_out_file = open(root + '/train_data.txt', 'w')
    val_out_file = open(root + '/val_data.txt', 'w')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', default='../data/', help='path to the folder containing data')
    args = parser.parse_args()
    enumerate_data(**args.__dict__)