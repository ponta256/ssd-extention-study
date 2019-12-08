import torch
import torchvision
import os
import cv2
import json
import numpy as np
import glob
import re


class BDDDetection(torchvision.datasets.vision.VisionDataset):
    CLASSES = ["bike",
               "car",
               "motor",
               "person",
               "rider",
               "traffic light",
               "traffic sign",
               "train",
               "truck",
               "bus"]
    BDD_ROOT = "/mnt/ssd/bdd100k/"

    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        super(BDDDetection, self).__init__(root, transforms, transform, target_transform)
        self.name = 'BDD'
        
        with open(BDDDetection.BDD_ROOT+"labels/bdd100k_labels_images_train.json") as f:
            self.anns = json.load(f)

    def __getitem__(self, index):
        ann = self.anns[index]
        img_path = os.path.join(self.BDD_ROOT+'images/100k/train/'+ann['name'])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        boxes = np.empty((0,4))
        labels = ann["labels"]
        ctgs = np.array([])
        for label in labels:
            if 'box2d' in label:
                ctg = int(BDDDetection.CLASSES.index(label["category"]))
                ctgs = np.append(ctgs, ctg)
                xmin = label['box2d']['x1']/w
                ymin = label['box2d']['y1']/h
                xmax = label['box2d']['x2']/w
                ymax = label['box2d']['y2']/h
                boxes = np.append(boxes, [[xmin, ymin, xmax, ymax]], axis=0)

        if self.transform is not None:
            img, boxes, ctgs = self.transform(img, boxes, ctgs)
            img = img[:, :, (2, 1, 0)]  # BGR -> RGB
            target = np.hstack((boxes, np.expand_dims(ctgs, axis=1)))
            # H,W,C -> C,H,W        
        return torch.from_numpy(img).permute(2, 0, 1), target, h, w

    def __len__(self):
        return len(self.anns)
