import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2

import argparse

labelmap = ["bike",
            "car",
            "motor",
            "person",
            "rider",
            "traffic light",
            "traffic sign",
            "train",
            "truck",
            "bus"]

color = [(255,255,0),
         (0,0,255),
         (128,128,128),
         (0,255,0),
         (128,128,0),
         (0,255,0),
         (255,0,0),
         (255,0,0),
         (0,128,0),
         (128,0,128)]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('image')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--use_pred_module', default=True, type=str2bool,
                    help='Use prediction module')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from fssd1280 import build_ssd

from data import config
cfg = config.bdd1280x768

num_classes = len(labelmap) + 1                      # +1 for background
net = build_ssd('test', cfg, args.use_pred_module)     # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()

print(net)

from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform

image = cv2.imread(args.image)

x = cv2.resize(image, (1280, 768)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy() # BGR -> RGB
x = torch.from_numpy(x).permute(2, 0, 1) # H,W,C -> C,H,W

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(image.shape[1::-1]).repeat(2)

for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= args.confidence_threshold:
        score = detections[0,i,j,0]
        label_name = labelmap[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        print(pt, score, label_name)
        cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color[i-1], 1)
        j+=1

cv2.imwrite('out.jpg', image)
