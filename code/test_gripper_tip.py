# python test.py --root-path /path/to/root/data/directory --use-normal --resume /path/to/model/file
# --exp-id 31 --save-dir /path/to/save/results

from data_loader import TrashPickerTestDataset
from model import UNet, ResNetUNet
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torch
from tensorboardX import SummaryWriter
from arguments import get_args
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from IPython import embed
import torch.nn as nn
from copy import deepcopy as copy

# setup argument formalities ===================================================
args = get_args()

# assign arguments
inp_size = (256, 512)
use_small_network = args.use_small_network
model_path = args.resume
save_dir = args.save_dir
root_path = args.root_path
stop_image_store = args.stop_image_store
use_resnet = args.use_resnet
prob_thr = [0.2,0.5,0.9]


# create result save dire ======================================================
for p in prob_thr:
    if not os.path.isdir(os.path.join(save_dir, str(p).replace('.','_'))):
        os.makedirs(os.path.join(save_dir, str(p).replace('.','_')))

# create file for storing the finger tip location ==============================
file_list = []
for p in prob_thr:
    file_list.append(open(os.path.join(save_dir, str(p).replace('.','_'), 'finger_tip_dist.txt'), 'w'))

# define the device ===========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define models ================================================================
if use_resnet:
    model = ResNetUNet(n_class=1)
else:
    model = UNet(inp_channel=3, num_classes=1, small_net=use_small_network)



# define data loader ===========================================================
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
normal_mean_std = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

test_joint_transform = joint_transforms.ComposeTest([
    joint_transforms.FreeScaleTest(inp_size)
])

test_input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)])

restore_transform = standard_transforms.Compose([
    extended_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()])

visualize = standard_transforms.ToTensor()

test_data_file_path = os.path.join(root_path, args.test_data_file_path)
test_dataset = TrashPickerTestDataset(test_data_file_path,
                                    root_path=root_path,
                                    joint_transform=test_joint_transform,
                                    transform=test_input_transform,
                                    crop_img = True)
test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# load the model ===============================================================
print("loading the model")
model.load_state_dict(torch.load(model_path))

# multi GPU ====================================================================
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model = model.to(device)

# divider mask
z = np.polyfit(np.array([558.59167816, 797.01748835]), np.array([447.19891699, 494.44119623]),1)
p = np.poly1d(z)
divider_mask = np.ones((1440, 1920)).astype(np.uint8)
mean_pt = [int(divider_mask.shape[0]/2 + divider_mask.shape[0]/20), int(divider_mask.shape[1]/2 + divider_mask.shape[0]/25)]
box_size = int(divider_mask.shape[0]/2)
divider_mask = divider_mask[int(mean_pt[0]-box_size/2):int(mean_pt[0]+box_size/2), int(mean_pt[1]-box_size/2):int(mean_pt[1]+box_size/2)]
Y, X = np.nonzero(divider_mask)

for x, y in zip(X, Y):
    if x - p(y) > -30:
        divider_mask[y,x] = 0
divider_mask_2 = np.zeros_like(divider_mask)
divider_mask_2[170:290, 100:570] = 1.0
# test =========================================================================
def test():
    prev_directory = len(prob_thr)*['']
    divider_mask_2 = len(prob_thr)*[None]
    divider_mask_2_rgb = len(prob_thr)*[None]
    model.eval()
    count = len(prob_thr)*[0]
    for j, data in enumerate(test_dataset_loader):
        with torch.no_grad():
            inp = data['img'].cuda() 
            pred_mask = model(inp)

            pred_prob = F.sigmoid(pred_mask)

            # for each prob threshold find the location and dump images into corresponding directorues
            for p_indx, p_thr in enumerate(prob_thr):
                pred_prob_bin = pred_prob > p_thr
                for i in range(pred_prob_bin.shape[0]):
                    bin_img = (255 * pred_prob_bin[i].data.cpu().numpy().astype(np.float32)).astype(np.uint8)[0,:,:]
                    bin_img = cv2.resize(bin_img, (720,720))

                    # get the divider mask based on the intial image
                    directory = os.path.join(*data['path'][i].split('/')[:-1])
                    if directory != prev_directory[p_indx]:
                        prev_directory[p_indx] = directory
                        contours,hierarchy = cv2.findContours(bin_img, 1, 2) 
                        c_index = np.argmax([cv2.contourArea(cnt) for cnt in contours])
                        hull = cv2.convexHull(contours[c_index])
                        divider_mask_2[p_indx] = np.zeros_like(bin_img)
                        cv2.fillConvexPoly(divider_mask_2[p_indx],hull,255)   
                        temp = np.expand_dims(divider_mask_2[p_indx], -1)
                        divider_mask_2_rgb[p_indx] = np.concatenate((temp,temp,temp),axis=2)

                    bin_img_pos = np.logical_and(np.logical_and(bin_img>0, divider_mask == 0), divider_mask_2[p_indx] > 0)
                    bin_img_neg = np.logical_and(np.logical_and(bin_img>0, divider_mask > 0), divider_mask_2[p_indx] > 0)
                     
                    # org_img
                    pil_image = restore_transform(inp[i].data.cpu())
                    org_image = np.array(pil_image)[:, :, ::-1]
                    org_image = cv2.resize(org_image, (720,720))
                    
                    try:
                        # find fingertip
                        Y, X = np.nonzero(bin_img_pos.astype(np.uint8))
                        indx = np.argmin(Y)
                        pos_p = {'x':X[indx],'y':Y[indx]}
                        
                        Y, X = np.nonzero(bin_img_neg.astype(np.uint8))
                        indx = np.argmin(Y)
                        neg_p = {'x':X[indx],'y':Y[indx]}

                        # draw circles
                        cv2.circle(org_image, (pos_p['x'], pos_p['y']), radius=50, color=(0), thickness=25)
                        cv2.circle(org_image, (neg_p['x'], neg_p['y']), radius=50, color=(0), thickness=25)
                        
                    except:
                        print("no fingertip found")

                    # store image
                    heatmap = cv2.resize(cv2.applyColorMap((255*pred_prob[i].data.cpu().numpy()[0]).astype(np.uint8), cv2.COLORMAP_JET), (720,720))
                    org_image = cv2.addWeighted(heatmap, 0.3, org_image, 0.7, 0)
                    org_image = cv2.addWeighted(divider_mask_2_rgb[p_indx], 0.2, org_image, 0.8, 0)
                    cv2.imwrite(os.path.join(save_dir, str(p_thr).replace('.','_'), '{:06d}.jpg'.format(count[p_indx])), org_image)
                    count[p_indx]+= 1
                    finger_tip_dist = np.sqrt((pos_p['x'] - neg_p['x'])**2 + (pos_p['y'] - neg_p['y'])**2)
                    file_list[p_indx].write(data['path'][i] + ' ' +  str(finger_tip_dist)+ '\n')
                    print("count = {}".format(count[p_indx]))

test()
for f in file_list:
    f.close()