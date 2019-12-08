from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.size = cfg['size']
        self.anchor_sizes = cfg['anchor_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        w = self.size[0]
        h = self.size[1]
        for k, f in enumerate(self.feature_maps):
            # for i, j in product(range(f), repeat=2):
            for i in range(f[0]):  # y
                for j in range(f[1]): # x
                    # print('W, H, X, Y', w, h, j, i)
                    # should be equal to f (e.g. 768/8=96)
                    f_k_x = w / self.steps[k]
                    f_k_y = h / self.steps[k]
                    # unit center x,y (normalized, e.g. 0.5 / 96)
                    cx = (j + 0.5) / f_k_x
                    cy = (i + 0.5) / f_k_y

                    s_k_x = self.anchor_sizes[k]/w
                    s_k_y = self.anchor_sizes[k]/h

                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_x/sqrt(ar), s_k_y*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
