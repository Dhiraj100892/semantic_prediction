from __future__ import print_function, division
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for suction challenge')
    parser.add_argument('--train-data-file-path', type=str, default='../data/train_hand_data.txt',
                        help='path to train file txt')
    parser.add_argument('--val-data-file-path', type=str, default='../data/val_hand_data.txt',
                        help='path to val file txt')
    parser.add_argument('--test-data-file-path', type=str, default='../data/test_data.txt',
                        help='path to test file txt')
    parser.add_argument('--root-path', default='',
                        help='path to the root directory for data (default: '')')
    parser.add_argument('--exp-id', type=int, required=True, help='name for storing the logs')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='learning rate for policy(default: 3e-3)')
    parser.add_argument('--pos-weight', type=float, default=10.0,
                        help='Weight on the positive data point (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for training (default: 16)')
    parser.add_argument('--log-dir', default='../logs_hand/',
                        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument('--save-dir', default='../results_hand/',
                        help='directory to save evaluation result in testing(useful while testing)')
    parser.add_argument('--save-iter', type=int, default=2000,
                        help='after how many iteration to save the data')
    parser.add_argument('--resume', type=str, default='../logs/0001/best_prec_model.pth',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='num of epochs to run the experiment')
    parser.add_argument('--use-small-network', action='store_true', default=False,
                        help='whether to use small network')
    parser.add_argument('--stop-random-rotation', action='store_true', default=False,
                        help='whether to do stop random rotation')
    parser.add_argument('--stop-jitter', action='store_true', default=False,
                        help='stop jitter')
    parser.add_argument('--stop-image-store', action='store_true', default=False,
                        help='stop storing images during visualization')
    parser.add_argument('--rot-angle', type=float, default=180.0,
                        help='the angles by which we can rotate image for data augmentation')
    parser.add_argument('--no-augmentation', action='store_true', default=False,
                        help='whether to do data augmentation during training ')
    parser.add_argument('--use_resnet', action='store_true', default=False,
                        help='use pretrained resent')
    parser.add_argument('--use_multigpu', action='store_true', default=False,
                        help='use multiple GPUS')
    parser.add_argument('--crop_img', action='store_true', default=False,
                        help='whether to crop images')    
    

    args = parser.parse_args()
    return args
