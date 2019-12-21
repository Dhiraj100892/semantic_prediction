# python test.py --root-path /path/to/root/data/directory --use-normal --resume /path/to/model/file
# --exp-id 31 --save-dir /path/to/save/results

from data_loader import SuctionTestDataset
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

# setup argument formalities ===================================================
args = get_args()

# assign arguments
inp_size = (256, 512)
use_small_network = args.use_small_network
model_path = args.resume
save_dir = args.save_dir
org_img_size = (1920, 1080)
root_path = args.root_path
stop_image_store = args.stop_image_store
log_dir = '{}{:04d}'.format(args.log_dir, args.exp_id)
print("log_dir = {}".format(log_dir))
writer = SummaryWriter(log_dir=log_dir)
use_resnet = args.use_resnet


# create result save dire ======================================================
grey_save_dir = os.path.join(save_dir, 'grey')
if not os.path.isdir(grey_save_dir):
    os.makedirs(os.path.join(save_dir, 'grey'))

color_save_dir = os.path.join(save_dir, 'color')
if not os.path.isdir(color_save_dir):
    os.makedirs(os.path.join(save_dir, 'color'))

heatmap_save_dir = os.path.join(save_dir, 'heatmap')
if not os.path.isdir(heatmap_save_dir):
    os.makedirs(os.path.join(save_dir, 'heatmap'))

# define models ================================================================
if use_resnet:
    model = ResNetUNet(n_class=1).cuda()
else:
    model = UNet(inp_channel=3, num_classes=1, small_net=use_small_network).cuda()

# define data loader ===========================================================
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
normal_mean_std = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
short_size = int(min(inp_size) / 0.875)

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
test_dataset = SuctionTestDataset(test_data_file_path,
                                  root_path=root_path,
                                  joint_transform=test_joint_transform,
                                  transform=test_input_transform)
test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

# load the model ===============================================================
print("loading the model")
model.load_state_dict(torch.load(model_path))


# test =========================================================================
def test():
    model.eval()
    val_visual = []
    for j, data in enumerate(test_dataset_loader):
        with torch.no_grad():
            inp = data['img'].cuda() 
            pred_mask = model(inp)

            pred_prob = F.sigmoid(pred_mask)

            # TODO: Need to add prediction mask visualization
            if not stop_image_store:
                for indx in range(inp.shape[0]):
                    pred_mask_heatmap = Image.fromarray(cv2.applyColorMap((255*pred_prob[indx].data.cpu().numpy()[0])
                                                                          .astype(np.uint8), cv2.COLORMAP_JET))
                    val_visual.extend([visualize(restore_transform(inp[indx].data.cpu())),
                                       visualize(pred_mask_heatmap)])

            # store the output
            pred_porb_bin = pred_prob > 0.5
            for i in range(pred_prob.shape[0]):
                temp_data = (255 * pred_porb_bin[i].data.cpu().numpy().astype(np.float32)).astype(np.uint8)[0,:,:]
                # grey
                temp_data_resized = cv2.resize(temp_data, org_img_size)
                grey_file_name = os.path.join(grey_save_dir, data['data_name'][i])
                cv2.imwrite(grey_file_name, temp_data_resized)

                # as of input
                color_out = np.zeros((org_img_size[1], org_img_size[0], 3)).astype(np.uint8)
                color_out[:,:,1] = temp_data_resized
                color_file_name = os.path.join(color_save_dir, data['data_name'][i])
                cv2.imwrite(color_file_name, color_out)

                # numpy array
                numpy_data = (255 * pred_prob[i].data.cpu().numpy()[0]).astype(np.uint8)
                numpy_data = cv2.resize(numpy_data, org_img_size).astype(np.float32) / 255.0

                # heatmap images
                org_img = cv2.imread(os.path.join(root_path, '/home/sawyer/projects/trash_picker/data/images_png', data['data_name'][i]))
                heatmap = cv2.resize(cv2.applyColorMap((255*numpy_data).astype(np.uint8), cv2.COLORMAP_JET),
                                     org_img_size)
                heatmap_file_name = os.path.join(heatmap_save_dir, data['data_name'][i])
                dst = cv2.addWeighted(org_img, 0.5, heatmap, 0.5, 0)
                cv2.imwrite(heatmap_file_name, dst)

    if not stop_image_store:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=2, padding=5)
        writer.add_image("val", val_visual)
        writer.add_image("val", val_visual)


test()

