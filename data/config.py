# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)


# SSD512 CONFIGS
# [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
vocd512 = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,        
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'box': [6, 6, 6, 6, 6, 4, 4],
    'aspect_ratios': [[2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

vocd512f = {
    'num_classes': 21,
    'lr_steps': (100, 130, 150),    
    'max_epoch': 150,  
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'box': [6, 6, 6, 6, 6, 4, 4],
    'aspect_ratios': [[2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# 1280x768(0), 640x384(1), 320x192(2), 160x96(3), 80x48(4), 40x24(5), 20x12(6), 10x6(7), 5x3(8)
bdd1280x768 = {
    'num_classes': 11,
    'lr_steps': (100, 130, 150),    
    'max_epoch': 150,
    # (w,h)    
    'size': (1280,768),
    # (h,w)
    'feature_maps': [(48,80), (24,40), (12,20), (6,10), (3,5)],
    'steps': [16, 32, 64, 128, 256],
    'anchor_sizes': [32, 64, 128, 256, 512],
    'box': [5, 5, 5, 5, 5],
    'aspect_ratios':  [(0.5,1.0,1.5,2.0,2.5),
                      (0.5,1.0,1.5,2.0,2.5),
                      (0.5,1.0,1.5,2.0,2.5),
                      (0.5,1.0,1.5,2.0,2.5),
                      (0.5,1.0,1.5,2.0,2.5)],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'BDD',
}

# SSD512 CONFIGS
# [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
voc512 = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,        
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# SSD300 CONFIGS
# 'lr_steps': (80000, 100000, 120000),
# 'max_iter': 120000,
voc = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,    
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
