import unittest
import cv2
import json
import torch

from data.anchor_genration import generate_anchor

def scale_img(img, gt, shortest_side=600):
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

class anchor_gneration_Test(unittest.TestCase):
    def setUp(self):
        file = "./data/easy/pic/TB1fO2wRVXXXXX3XVXXXXXXXXXX_!!0-item_pic_main_image_986346.jpg"
        img = cv2.imread(file)
        with open('./data/easy/ocr.json', 'r', encoding='utf8') as f:
            ocr = json.load(f)
        gt = ocr['TB1fO2wRVXXXXX3XVXXXXXXXXXX_!!0-item_pic_main_image_986346.jpg']['Bbox']
        self.img, self.s_gt = scale_img(img, gt)

    def generate_anchor(self):
        img = torch.from_numpy(self.img)
        img = torch.squeeze(img, 0)
        for box in self.s_gt:
            result = generate_anchor(img, box)
            print(result)
