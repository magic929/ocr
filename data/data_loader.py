import os
import json

import cv2
import numpy as np
from torch.utils.data import Dataset

class OCRD(Dataset):
    def __init__(self, root, json_file):
        self.root = root
        self.pics = os.listdir(self.root)
        with open(json_file, 'r', encoding='utf8') as f:
            self.ocr = json.load(f)

    def __getitem__(self, index):
        filename = self.root + self.pics[index]
        img = cv2.imread(filename)
        gt = self.ocr[self.pics[index]]['Bbox']
        img, s_gt = self.scale_img(img, gt)

        return img, s_gt, filename
    
    def __len__(self):
        return len(self.pics)
    
    def scale_img(self, img, gt, shortest_side=600):
        height = img.shape[0]
        width = img.shape[1]
        scale = float(shortest_side) / float(min(height, width))
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if img.shape[0] < img.shape[1] and img.shape[0] != 600:
            img = cv2.resize(img, (600, img.shape[1]))
        elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
            img = cv2.resize(img, (img.shape[0], 600))
        elif img.shape[0] != 600:
            img = cv2.resize(img, (600, 600))

        h_scale = float(img.shape[0]) / float(height)
        w_scale = float(img.shape[1]) / float(width)
        
        scale_gt = []
        for box in gt:
            scale_box = []
            for i in range(len(box)):
                if i % 2 == 0:
                    scale_box.append(int(int(box[i]) * w_scale))
                else:
                    scale_box.append(int(int(box[i]) * h_scale))
            
            scale_gt.append(scale_box)
        
        return img, scale_gt
        

        
        
