import time
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch import optim

import evaluate
from model import ctpn, utils, LOSS
from data.data_loader import OCRD
from data.anchor_genration import generate_anchor, tag_anchor
from utils.logger import Logger, logging

train_logger = Logger('./output/trian.log', 'train', logging.DEBUG)

no_grad = [
    'cnn.0.weight',
    'cnn.0.bias',
    'cnn.2.weight',
    'cnn.2.bias'
    ]

using_cuda = True
epoch = 50
epoch_change = 20
lr_front = 0.001
lr_behind = 0.0001
display_iter = 1400
val_iter = 2900
save_epchoe = 10
MODEL_SAVE_PATH = './model'
val_batch_size = 20


def train():
    net = ctpn.CTPN()
    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    
    utils.init_weight(net)

    if using_cuda:
        net.cuda()
    
    net.train()

    criterion = LOSS.CTPN_Loss(using_cuda=using_cuda)

    full_data = OCRD('./data/easy/pic/', './data/easy/ocr.json')
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_loader = DataLoader(train_data, 1, shuffle=True)
    val_loader = DataLoader(val_data, 1, shuffle=True)
    
    trian_loss_list = []
    test_loss_list = []

    for i in range(epoch):
        if i > epoch_change:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        iteration = 0
        total_loss = 0
        total_cls_loss = 0
        total_v_reg_loss = 0
        total_o_reg_loss = 0
        total_iter = len(train_loader)
        start = time.time()
        for img, tag, filename in train_loader:
            # img_pil = Image.open(filename[0])
            tensor_img = img.permute((0, 3, 1, 2))
            img = torch.squeeze(img, 0)
            if using_cuda:
                tensor_img = tensor_img.to(dtype=torch.float)
            else:
                tensor_img = tensor_img.to(dtype=torch.float)
            
            vertical_pred, score, side_refinement = net(tensor_img)
            del tensor_img

            positive = []
            negative = []
            vertical_reg = []
            side_refinement_reg = []

            try:
                for box in tag:
                    gt_anchor = generate_anchor(img, box)
                    positive1, negative1, vertical_reg1, side_refinement_reg1 = tag_anchor(gt_anchor, score, box)
                    positive += positive1
                    negative += negative1
                    vertical_reg += vertical_reg1
                    side_refinement_reg += side_refinement_reg1
                
            except Exception as e:
                train_logger.warn("the error is %s" %e)
                train_logger.warn("warning: img %s raise error" % filename)
                iteration += 1
                exit()
                # continue
            
            if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
                iteration += 1
                continue

            optimizer.zero_grad()
            loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive, negative, vertical_reg, side_refinement_reg)
            loss.backward()
            optimizer.step()
            iteration += 1
            total_loss += float(loss)
            total_cls_loss += float(cls_loss)
            total_v_reg_loss += float(v_reg_loss)
            total_o_reg_loss += float(o_reg_loss)

            if iteration % display_iter == 0:
                end = time.time()
                total_time = start - end
                print('Epoch: {2}/{3}, Iteration: {0}/{1}, loss: {4}, cls_loss: {5}, v_reg_loss: {6}, o_reg_loss: {7}, {8}'.
                      format(iteration, total_iter, i, epoch, total_loss / display_iter, total_cls_loss / display_iter,
                             total_v_reg_loss / display_iter, total_o_reg_loss / display_iter, filename))

                trian_loss_list.append(total_loss)

                total_loss = 0
                total_cls_loss = 0
                total_v_reg_loss = 0
                total_o_reg_loss = 0
                start = time.time()
            
            if iteration % val_iter == 0:
                net.eval()
                train_logger.info("start evaluate at {} epoch {} iteration".format(i, iteration))
                val_loss = evaluate.val(net, criterion, val_batch_size, using_cuda, train_logger, val_loader)
                train_logger.info('End evaluate.')
                net.train()
                start_time = time.time()
                test_loss_list.append(val_loss)
                
            if i % save_epchoe == 0:
                print('Model saved at ./output/ctpn-{0}-{1}.mode'.format(i, iteration))
                torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-{1}.model'.format(i, iteration)))
        
        print('Model saved at ./output/ctpn-{0}-end.model'.format(i))
        torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-end.model'.format(i)))


if __name__ == "__main__":
    train()
        