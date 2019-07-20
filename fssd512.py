import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os

class FSSD(nn.Module):

    def fuse_module(self, in_ch, size):
        fm = [
            nn.Conv2d(in_ch, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        ]
        return fm

    def fe_module(self, in_ch, out_ch):
        fe = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        return fe
    
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

    def __init__(self, phase, cfg, use_pred_module=False):
        super(FSSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.cfg = cfg
        
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
            # pool5を2×2―s2から3×3―s1に変更し，
            # à trousアルゴリズムをholes"を埋めるために使用
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
        ]


        # Fusion Module
        size = 64
        self.fm0 = nn.ModuleList(self.fuse_module(512, size))          # 4_3  # 64 (OUT), 512
        self.fm1 = nn.ModuleList(self.fuse_module(1024, size))         # fc7  # 32 (OUT), 1024
        self.fm2 = nn.ModuleList(self.fuse_module(512, size))          # 6_2  # 16 (OUT), 512
        self.fm = [self.fm0, self.fm1, self.fm2]
        self.fm_bn = nn.BatchNorm2d(256*3, affine=True)


        source = [512, 1024, 512, 256, 256, 256, 256]
        box = cfg['box']
        
        # Feature Extractor
        self.fe0 = nn.ModuleList(self.fe_module(256*3, source[0]))
        self.fe1 = nn.ModuleList(self.fe_module(source[0], source[1]))
        self.fe2 = nn.ModuleList(self.fe_module(source[1], source[2]))
        self.fe3 = nn.ModuleList(self.fe_module(source[2], source[3]))
        self.fe4 = nn.ModuleList(self.fe_module(source[3], source[4]))
        self.fe5 = nn.ModuleList(self.fe_module(source[4], source[5]))
        self.fe6 = nn.ModuleList(self.fe_module(source[5], source[6])) # check, k4, p1
        self.fe = [self.fe0, self.fe1, self.fe2, self.fe3, self.fe4, self.fe5, self.fe6]

        # prediction module
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
            nn.Conv2d(source[0], box[0]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[1], box[1]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[2], box[2]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[3], box[3]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[4], box[4]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[5], box[5]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[6], box[6]*self.num_classes, kernel_size=3, padding=1)
        ]
        
        # NEED TO CHECK THIS
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        sources.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
        sources.append(x) 

        upsampled = list()
        for k, v in enumerate(self.fm):
            x = sources[k]
            for l in v:
                x = l(x)
                x = F.interpolate(x, 64, mode='bilinear', align_corners=False)
            upsampled.append(x)
        fused_feature = torch.cat(upsampled, 1)
        x = self.fm_bn(fused_feature)

        feature_maps = list()
        for l in self.fe[0]:
            x = l(x)
        feature_maps.append(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        for v in self.fe[1:-1]:
            for l in v:
                x = l(x)
            feature_maps.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        for l in self.fe[-1]:
            x = l(x)
        feature_maps.append(x)            

        # apply multibox head to source layers
        if self.use_pred_module:
            for (x, p, l, c) in zip(feature_maps, self.pm, self.loc, self.conf):
                xs = p[7](x)
                for i in range(0, 6):
                    x = p[i](x)
                x = xs + p[6](x)
                x = p[8](x)
                x = p[9](x)
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(feature_maps, self.loc, self.conf):
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
    return FSSD(phase, cfg, use_pred_module)
