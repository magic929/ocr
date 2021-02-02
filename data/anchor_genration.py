import math

import numpy as np
from PIL import Image, ImageDraw


# def generate_basic_anchors(sizes, base_size=16):
#     base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
#     anchors = np.zeros((len(sizes), 4), np.int32)
    
#     for index, (h, w) in enumerate(sizes):
#         anchors[index] = scale_anchor(base_anchor, h, w)
    
#     return anchors
    

# def scale_anchor(anchor, h, w):
#     x = (anchor[0] + anchor[2]) / 2
#     y = (anchor[1] + anchor[3]) / 2
#     scaled_anchor = anchor.copy()
#     scaled_anchor[0] = x - w / 2
#     scaled_anchor[1] = y - h / 2
#     scaled_anchor[2] = x + w / 2
#     scaled_anchor[3] = y + h / 2

#     return scaled_anchor


def generate_anchor(img, box, anchor_width=16):
    result = []
    left_anchors = int(math.floor(max(min(box[0], box[6]), 0) / anchor_width))
    right_anchors = int(math.ceil(min(max(box[2], box[4]), img.shape[1]) / anchor_width))

    if right_anchors * 16 + 15 > img.shape[1]:
        right_anchors -= 1
    
    positions = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchors, right_anchors)]
    tops, bottoms = top_bottom(img, positions, box)
    for i in range(len(positions)):
        position = int(positions[i][0] / anchor_width)
        h = bottoms[i] - tops[i] + 1
        cy = (float(bottoms[i]) + float(tops[i])) / 2.0
        result.append((position, cy, h))
    
    return result


def top_bottom(img, positions, box):
    height, width = img.shape[0], img.shape[1]
    black = Image.new('RGB', (width, height), (0, 0, 0))
    tops = []
    bottoms = []
    top_flag = False
    bottom_flag = False

    draw = ImageDraw.Draw(black)
    draw.polygon(list(zip(box[::2], box[1::2])), outline=(255, 0, 0))

    black = np.array(black)

    for k in range(len(positions)):
        for y in range(0, height - 1):
            for x in range(positions[k][0], positions[k][1] + 1):
                if black[y, x, 0] == 255:
                    tops.append(y)
                    top_flag = True
                    break

            if top_flag == True:
                break

        for y in range(height - 1, -1 , -1):
            for x in range(positions[k][0], positions[k][1] + 1):
                if black[y, x, 0] == 255:
                    bottoms.append(y)
                    bottom_flag = True
                    break
            
            if bottom_flag == True: 
                break
        
        top_flag = False
        bottom_flag = False

    return tops, bottoms


def cal_IoU(cy1, h1, cy2, h2):
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    top_min = min(y_top1, y_top2)
    bottom_max = max(y_bottom1, y_bottom2)
    union = bottom_max - top_min + 1
    intersection = h1 + h2 - union
    iou = float(intersection) / float(union)
    if iou < 0:
        return 0.0
    else: 
        return iou


def cal_y(cy, h):
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)

    return y_top, y_bottom


def valid_anchor(cy, h, height):
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    
    if bottom > (height * 16 - 1):
        return False
    
    return True


def tag_anchor(gt_anchor, cnn_out, gt_box):
    anchor_height = [11, 16, 22, 32, 46, 66, 95, 134, 191, 273]
    height = cnn_out.shape[2]
    width = cnn_out.shape[3]
    positives = []
    negatives = []
    vertical_regs = []
    side_refinement_reg = []
    x_left = min(gt_box[0], gt_box[6])
    x_right = max(gt_box[2], gt_box[4])
    left_side = False
    right_side = False

    for a in gt_anchor:
        if a[0] >= int(width - 1):
            continue

        if x_left in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        if x_right in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False
        
        iou = np.zeros((height, len(anchor_height)))
        temp_positives = []
        
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = cal_IoU((float(i) * 16 + 7.5), anchor_height[j], a[1], a[2])
                if iou[i][j] > 0.7:
                    temp_positives.append((a[0], i, j, iou[i][j]))
                    if left_side:
                        o = (float(x_left) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                    
                    if right_side:
                        o = (float(x_right) - (float(a[0] * 16.0 + 7.5))) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                
                if iou[i][j] < 0.5:
                    negatives.append((a[0], i, j, iou[i][j]))
                
                if iou[i][j] > 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    vertical_regs.append((a[0], i, j, vc, vh, iou[i][j]))
            
            if len(temp_positives) == 0:
                max_position = np.where(iou == np.max(iou))
                temp_positives.append((a[0], max_position[0][0], max_position[1][0], np.max(iou)))

                if left_side:
                    o = (float(x_left) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                    side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
                
                if right_side:
                    o = (float(x_left) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                    side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
                
                if np.max(iou) <= 0.5:
                    vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                    vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                    vertical_regs.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))

        positives += temp_positives
    
    return positives, negatives, vertical_regs, side_refinement_reg