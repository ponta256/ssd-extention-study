import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc512
import os

class SSD(nn.Module):

    def pred_module(self, in_ch):
        
        pm =[
            nn.Conv2d(in_ch, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),            
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),            
            nn.ReLU(inplace=True),            
            nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            nn.Conv2d(in_ch, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        ]
        return pm

    def __init__(self, phase, size, num_classes, use_pred_module=False):
        super(SSD, self).__init__()

        base = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 1_1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 1_2  # 512
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 2_1
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 2_2  # 256
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 3_1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 3_2                  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 3_3  # 128
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 4_1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 4_2                  
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 4_3  # 64 (OUT)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 5_1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 5_2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 5_3
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # pool5
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True), # fc6
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True), # fc7  # 32 (OUT)
        ]

        extras = [
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # 6_1
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), # 6_2    # 16 OUT 
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # 7_1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # 7_2     # 8 OUT
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # 8_1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # 8_2     # 4 OUT
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # 9_1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # 9_2     # 2 OUT
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # 10_1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # 10_2    # 1 OUT
        ]

        box = [4, 6, 6, 6, 6, 4, 4]            
        source = [512, 1024, 512, 256, 256, 256, 256]            

        self.use_pred_module = use_pred_module
        if self.use_pred_module:
            self.pm0 = nn.ModuleList(self.pred_module(source[0]))
            self.pm1 = nn.ModuleList(self.pred_module(source[1]))
            self.pm2 = nn.ModuleList(self.pred_module(source[2]))
            self.pm3 = nn.ModuleList(self.pred_module(source[3]))
            self.pm4 = nn.ModuleList(self.pred_module(source[4]))
            self.pm5 = nn.ModuleList(self.pred_module(source[5]))
            self.pm6 = nn.ModuleList(self.pred_module(source[6]))
            self.pm = [self.pm0, self.pm1, self.pm2, self.pm3, self.pm4, self.pm5, self.pm6]
            source = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
        
        loc_layers = [
            nn.Conv2d(source[0], box[0]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[1], box[1]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[2], box[2]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[3], box[3]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[4], box[4]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[5], box[5]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[6], box[6]*4, kernel_size=3, padding=1),
        ]
            
        conf_layers = [
            nn.Conv2d(source[0], box[0]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[1], box[1]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[2], box[2]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[3], box[3]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[4], box[4]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[5], box[5]*num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[6], box[6]*num_classes, kernel_size=3, padding=1)
        ]
        
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc512

        # NEED TO CHECK THIS
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3        
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)
        # print(s.shape)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        # print(x.shape)        

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k in [5, 11, 17, 23, 29]:
                sources.append(x)        
                # print(x.shape)
            
        # apply multibox head to source layers
        if self.use_pred_module:
            for (x, p, l, c) in zip(sources, self.pm, self.loc, self.conf):
                '''
                xs = p[3](x)
                x = p[0](x)
                x = p[1](x)
                x = xs + p[2](x)
                '''
                xs = p[7](x)
                for i in range(0, 6):
                    x = p[i](x)
                x = xs + p[6](x)
                x = p[8](x)
                x = p[9](x)
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print('CF', conf.shape)        

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors                
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            for c in torch.load(base_file, map_location=lambda storage, loc: storage):
                print(c)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_ssd(phase, cfg, use_pred_module=False):
    if phase != "test" and phase != "train" and phase != 'stat':
        print("ERROR: Phase: " + phase + " not recognized")
        return
    return SSD(phase, cfg['min_dim'], cfg['num_classes'], use_pred_module)
