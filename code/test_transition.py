# python main.py --pos-weight 11 --epoch 400 --exp-id 1 --lr 1e-3 --rot-angle 180.0
# TODO:
# 1) Include denseCRF while evaluating

import torch.nn as nn
from torch import optim
from data_loader import TransitionTestDataset
from model import UNet, ResNetUNet
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torch
from termcolor import colored
from tensorboardX import SummaryWriter
from arguments import get_args
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import random
import time
import torchvision.models as models
from IPython import embed
import os

# setup argument formalities ===================================================
args = get_args()

# assign arguments
lr = args.lr
num_epoch = args.epochs
save_iter = args.save_iter
log_dir = '{}{:04d}'.format(args.log_dir, args.exp_id)
print("log_dir = {}".format(log_dir))
stop_image_store = args.stop_image_store
stop_jitter = args.stop_jitter
stop_random_rotation = args.stop_random_rotation
use_small_network = args.use_small_network
rot_angle = args.rot_angle
no_augmentation = args.no_augmentation
train_data_file_path = args.train_data_file_path
val_data_file_path = args.val_data_file_path
root_path = args.root_path
use_resnet = args.use_resnet
use_multigpu = args.use_multigpu
crop_img = args.crop_img
model_path = args.resume
test_data_file_path = os.path.join(root_path, args.test_data_file_path)

# create out file ==============================================================
out_file = open('stack_prediction_{}.txt'.format(model_path.split('/')[-2]), 'w')

# define the device ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define models ================================================================
model = models.resnet18(pretrained=True)

# change last layer ============================================================
model.fc = nn.Linear(in_features=512, out_features=1)

# load the model ===============================================================
print("loading the model")
model.load_state_dict(torch.load(model_path))

# multi GPU ====================================================================
if use_multigpu:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

model = model.to(device)

# define data loader ===========================================================
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

test_input_transform = standard_transforms.Compose([
    standard_transforms.Resize((224,224)),
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)])

test_dataset = TransitionTestDataset(test_data_file_path,
                                root_path=root_path,
                                transform=test_input_transform)
test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

# validation ===================================================================
def test():
    model.eval()
    loop_start = time.time()
    for j, data in enumerate(test_dataset_loader):
        print('iteration = {}'.format(j))
        with torch.no_grad():
            inp = data['img'].to(device)
            path = data['path']
            pred_mask = model(inp)
            pred_prob = F.sigmoid(pred_mask)
            pred_mask_flat = pred_prob.view(-1)
            for i, l in enumerate(pred_mask_flat):
                out_file.write(path[i] + ' ' + str(float(l.data.cpu().item())) + '\n')

test()

out_file.close()