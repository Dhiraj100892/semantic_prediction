from model import UNet, ResNetUNet
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
from copy import deepcopy as copy

class FingerTipPred():
    def __init__(self, model_path, inp_size = (256,512)):

        # define the device ===========================================================
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define models ================================================================
        self.model = UNet(inp_channel=3, num_classes=1, small_net=False)
        # load the model
        self.model.load_state_dict(torch.load(model_path))

        # load the model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # params
        self.inp_size = inp_size

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_prediction(self, img, prob_thr = 0.5):
        """
        img --> in RGB format[0-255]
        """
        # crop the image
        org_img = copy(img)
        mean_pt = [int(img.shape[0]/2 + img.shape[0]/20), int(img.shape[1]/2 + img.shape[0]/25)]
        box_size = int(img.shape[0]/2)
        img = img[int(mean_pt[0]-box_size/2):int(mean_pt[0]+box_size/2), int(mean_pt[1]-box_size/2):int(mean_pt[1]+box_size/2)]
    
        # convert image to PIL Image
        im = Image.fromarray(np.uint8(img))
        inp_size = im.size

        # resize image
        im = im.resize(self.inp_size, Image.BILINEAR)

        # create a mini-batch as expected by the model
        inp = self.preprocess(im)
        inp = inp.unsqueeze(0)
        inp = inp.to(self.device)
        with torch.no_grad():
            pred_mask = self.model(inp)
            pred_prob = F.sigmoid(pred_mask)
            pred_prob_bin = pred_prob > prob_thr
        
        bin_img = (255 * pred_prob_bin[0].data.cpu().numpy().astype(np.float32)).astype(np.uint8)[0,:,:]
        bin_img = cv2.resize(bin_img, inp_size)
        
        # numpy array
        numpy_data = (255 * pred_prob[0].data.cpu().numpy()[0]).astype(np.uint8)
        numpy_data = cv2.resize(numpy_data, inp_size).astype(np.float32) / 255.0
        heatmap = cv2.resize(cv2.applyColorMap((255*numpy_data).astype(np.uint8), cv2.COLORMAP_JET),
                                     inp_size)
        heatmap = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        # find the finger tip ######################### 
        Y, X = np.nonzero(bin_img)
        
        # input line equation
        z = np.polyfit(np.array([558.59167816, 797.01748835]), np.array([447.19891699, 494.44119623]),1)

        # find the points on one side of line find the tip points
        p = np.poly1d(z)

        pos_p = None
        neg_p = None
        for x, y in zip(X, Y):
            if x - p(y) > 0:
                #img[y][x] = 0
                if pos_p is None or pos_p['y'] > y:
                    pos_p = {'x':x, 'y':y}
            else:
                if neg_p is None or neg_p['y'] > y:
                    neg_p = {'x':x, 'y':y}

        cv2.circle(heatmap, (pos_p['x'], pos_p['y']), radius=50, color=(0), thickness=25)
        cv2.circle(heatmap, (neg_p['x'], neg_p['y']), radius=50, color=(0), thickness=25)

        cv2.circle(org_img, (int(mean_pt[1]-box_size/2) + pos_p['x'], int(mean_pt[0]-box_size/2) + pos_p['y']), radius=50, color=(0), thickness=25)
        cv2.circle(org_img, (int(mean_pt[1]-box_size/2) + neg_p['x'], int(mean_pt[0]-box_size/2) + neg_p['y']), radius=50, color=(0), thickness=25)

        return {'finger_1': (int(mean_pt[1]-box_size/2) + pos_p['x'], int(mean_pt[0]-box_size/2) + pos_p['y']), 'finger_2': (int(mean_pt[1]-box_size/2) + neg_p['x'], int(mean_pt[0]-box_size/2) + neg_p['y'])}, heatmap, org_img
