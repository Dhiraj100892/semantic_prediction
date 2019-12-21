# python main.py --pos-weight 11 --epoch 400 --exp-id 1 --lr 1e-3 --rot-angle 180.0
# TODO:
# 1) Include denseCRF while evaluating

import torch.nn as nn
from torch import optim
from data_loader import SuctionDataset
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
# setup argument formalities ===================================================
args = get_args()

# assign arguments
lr = args.lr
num_epoch = args.epochs
save_iter = args.save_iter
log_dir = '{}{:04d}'.format(args.log_dir, args.exp_id)
print("log_dir = {}".format(log_dir))
writer = SummaryWriter(log_dir=log_dir)
inp_size = (256, 512)
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

# write hyper params to file
args_dict = vars(args)
arg_file = open(log_dir + '/args.txt', 'w')
for arg_key in args_dict.keys():
    arg_file.write(arg_key + " = {}\n".format(args_dict[arg_key]))
arg_file.close()

if not stop_image_store:
    import cv2

# define models ================================================================
if use_resnet:
    model = ResNetUNet(n_class=1).cuda()
else:
    model = UNet(inp_channel=3, num_classes=1, small_net=use_small_network).cuda()

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]).float().cuda())

# define optimizer =============================================================
optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=0.0005)

# define data loader ===========================================================
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
normal_mean_std = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
short_size = int(min(inp_size) / 0.875)

if stop_random_rotation:
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomCrop(inp_size)])
else:
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.RandomRotate(rot_angle),
        joint_transforms.RandomCrop(inp_size)])

val_joint_transform = joint_transforms.Compose([
    joint_transforms.FreeScale(inp_size)
])

if stop_jitter:
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)])
else:
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)])

val_input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)])

restore_transform = standard_transforms.Compose([
    extended_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()])

visualize = standard_transforms.ToTensor()

if no_augmentation:
    train_dataset = SuctionDataset(train_data_file_path,
                                   root_path=root_path,
                                   joint_transform=val_joint_transform,
                                   transform=val_input_transform)
else:
    train_dataset = SuctionDataset(train_data_file_path,
                                   root_path=root_path,
                                   joint_transform=train_joint_transform,
                                   transform=train_input_transform)

val_dataset = SuctionDataset(val_data_file_path,
                             root_path=root_path,
                             joint_transform=val_joint_transform,
                             transform=val_input_transform)

train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataset_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


# define metrics for the progress ==============================================
def cal_performance(pred_label, gt_label):
    """

    :param pred_label: torch Tensor
    :param gt_label: torch Tensor
    :return: prec_0, prec_1, f1_0, f1_1, accu_0, accu_1
    """
    pred_0 = (pred_label == 0.0).float()
    gt_0 = (gt_label == 0.0).float()
    true_pos = torch.sum(torch.dot(pred_label, gt_label)) 
    false_pos = torch.sum(pred_label) - true_pos 
    true_neg = torch.sum(torch.dot(pred_0, gt_0)) 
    false_neg = torch.sum(pred_0) - true_neg 
    accu_0 = true_neg / (false_pos + true_neg)
    accu_1 = true_pos / (true_pos + false_neg)
    prec_0 = true_neg / (true_neg + false_neg) 
    prec_1 = true_pos / (true_pos + false_pos) 
    f1_0 = 2 * (prec_0.item() * accu_0.item()) / (prec_0.item() + accu_0.item())
    f1_1 = 2 * (prec_1.item() * accu_1.item()) / (prec_1.item() + accu_1.item())
 
    return prec_0.item(), prec_1.item(), f1_0, f1_1, accu_0.item(), accu_1.item()


# train ========================================================================
def train(epoch):
    model.train()
    global iter_num
    train_visual = []
    vis = False
    for j, data in enumerate(train_dataset_loader):
        start_time = time.time()
        inp = data['img'].cuda()
        gt_mask = data['mask'].cuda()

        pred_mask = model(inp)

        gt_mask_flat = gt_mask.view(-1)
        pred_mask_flat = pred_mask.view(-1)

        loss = criterion(pred_mask_flat, gt_mask_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.random() < 0.005 and not vis and not stop_image_store:
            vis = True
            pred_prob = F.sigmoid(pred_mask)
            # prediction mask visualization
            for indx in range(inp.shape[0]):
                gt_mask_heatmap = Image.fromarray(cv2.applyColorMap((255*gt_mask[indx].data.cpu().numpy()[0])
                                                                    .astype(np.uint8), cv2.COLORMAP_JET))
                pred_mask_heatmap = Image.fromarray(cv2.applyColorMap((255*pred_prob[indx].data.cpu().numpy()[0])
                                                                      .astype(np.uint8), cv2.COLORMAP_JET))
                train_visual.extend([visualize(restore_transform(inp[indx].data.cpu())),
                                   visualize(gt_mask_heatmap),
                                   visualize(pred_mask_heatmap)])

            train_visual = torch.stack(train_visual, 0)
            train_visual = vutils.make_grid(train_visual, nrow=3, padding=5)
            writer.add_image("train/epoch_{}/iter_{}".format(epoch, iter_num), train_visual)

        # add visualization
        writer.add_scalar('train/loss', loss.item(), iter_num)

        # performance metric
        pred_label = (pred_mask_flat > 0).float()
        prec_0, prec_1, f1_0, f1_1, accu_0, accu_1 = cal_performance(pred_label, gt_mask_flat)

        writer.add_scalar('train/label_1_prec', prec_1, iter_num)
        writer.add_scalar('train/label_0_prec', prec_0, iter_num)
        writer.add_scalar('train/label_1_f1', f1_1, iter_num)
        writer.add_scalar('train/label_0_f1', f1_0, iter_num)
        writer.add_scalar('train/label_1_accuracy', accu_0, iter_num)
        writer.add_scalar('train/label_0_accuracy', accu_1, iter_num)

        iter_num += 1
        print("epoch = {} iteration = {} loss = {} time = {}s".format(epoch, iter_num, loss.item(), time.time()-start_time))

        # model saving code as well
        if iter_num % save_iter == 0:
            torch.save(model.state_dict(),
                       log_dir + '/model_{}.pth'.format(iter_num))


# validation ===================================================================
def val(epoch):
    model.eval()
    loss_list = []
    label_1_prec_list = []
    label_0_prec_list = []
    label_1_f1_list = []
    label_0_f1_list = []
    label_1_accuracy_list = []
    label_0_accuracy_list = []
    val_visual = []
    global best_val_prec
    global best_val_f1
    global best_val_loss

    loop_start = time.time()
    for j, data in enumerate(val_dataset_loader):
        with torch.no_grad():
            inp = data['img'].cuda()
            gt_mask = data['mask'].cuda()

            pred_mask = model(inp)

            gt_mask_flat = gt_mask.view(-1)
            pred_mask_flat = pred_mask.view(-1)

            # loss
            loss = criterion(pred_mask_flat, gt_mask_flat)

            # metric
            pred_label = (pred_mask_flat > 0).float()
            prec_0, prec_1, f1_0, f1_1, accu_0, accu_1 = cal_performance(pred_label, gt_mask_flat)

            # append results
            loss_list.append(loss.item())
            label_1_prec_list.append(prec_1)
            label_0_prec_list.append(prec_0)
            label_1_f1_list.append(f1_1)
            label_0_f1_list.append(f1_0)
            label_1_accuracy_list.append(accu_1)
            label_0_accuracy_list.append(accu_0)

            # add prediction mask visualization
            if not stop_image_store:
                pred_prob = F.sigmoid(pred_mask)
                for indx in range(inp.shape[0]):
                    gt_mask_heatmap = Image.fromarray(cv2.applyColorMap((255*gt_mask[indx].data.cpu().numpy()[0])
                                                                        .astype(np.uint8), cv2.COLORMAP_JET))
                    pred_mask_heatmap = Image.fromarray(cv2.applyColorMap((255*pred_prob[indx].data.cpu().numpy()[0])
                                                                          .astype(np.uint8), cv2.COLORMAP_JET))
                    val_visual.extend([visualize(restore_transform(inp[indx].data.cpu())),
                                       visualize(gt_mask_heatmap),
                                       visualize(pred_mask_heatmap)])

    print(colored("epoch = {} Validation loss = {} class_0_accuracy = {} class_1_accuracy = {} time taken = {}s".
                  format(epoch, np.mean(loss_list), np.mean(label_0_accuracy_list),
                         np.mean(label_1_accuracy_list), time.time()-loop_start), 'green'))

    if not stop_image_store:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image("val/epoch_{}".format(epoch), val_visual)

    loss_list = np.array(loss_list)
    loss_list[np.isnan(loss_list)] = 0
    writer.add_scalar('val/loss', np.mean(loss_list), epoch)

    label_1_prec_list = np.array(label_1_prec_list)
    label_1_prec_list[np.isnan(label_1_prec_list)] = 0
    writer.add_scalar('val/label_1_prec', np.mean(label_1_prec_list), epoch)

    label_0_prec_list = np.array(label_0_prec_list)
    label_0_prec_list[np.isnan(label_0_prec_list)] = 0
    writer.add_scalar('val/label_0_prec', np.mean(label_0_prec_list), epoch)

    label_1_f1_list = np.array(label_1_f1_list)
    label_1_f1_list[np.isnan(label_1_f1_list)] = 0
    writer.add_scalar('val/label_1_f1', np.mean(label_1_f1_list), epoch)

    label_0_f1_list = np.array(label_0_f1_list)
    label_0_f1_list[np.isnan(label_0_f1_list)] = 0
    writer.add_scalar('val/label_0_f1', np.mean(label_0_f1_list), epoch)

    label_1_accuracy_list = np.array(label_1_accuracy_list)
    label_1_accuracy_list[np.isnan(label_1_accuracy_list)] = 0
    writer.add_scalar('val/label_1_accuracy', np.mean(label_1_accuracy_list), epoch)

    label_0_accuracy_list = np.array(label_0_accuracy_list)
    label_0_accuracy_list[np.isnan(label_0_accuracy_list)] = 0
    writer.add_scalar('val/label_0_accuracy', np.mean(label_0_accuracy_list), epoch)

    # store model
    # model saving code as well
    if best_val_prec < np.mean(label_1_prec_list):
        torch.save(model.state_dict(), log_dir + '/best_prec_model.pth')
        best_val_prec = np.mean(label_1_prec_list)

    if best_val_f1 < np.mean(label_1_f1_list):
        torch.save(model.state_dict(), log_dir + '/best_f1_model.pth')
        best_val_f1 = np.mean(label_1_f1_list)

    if best_val_loss > np.mean(loss_list):
        torch.save(model.state_dict(), log_dir + '/best_loss_model.pth')
        best_val_loss = np.mean(loss_list)


iter_num = 0
best_val_prec = 0.0
best_val_f1 = 0.0
best_val_loss = float('inf')
for i in range(num_epoch):
    with torch.no_grad():
        val(i)
    train(i)

